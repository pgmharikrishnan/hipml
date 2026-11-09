//! Move a prefilled single-sequence state into a batch slot.
//!
//! The serving loop prefills on `GpuEngine` because that is the engine with the
//! chunked-GEMM path and the prefix cache. Once a prompt is complete, its KV
//! rows and SSM recurrent state are copied device-to-device into one slot of
//! the batch engine, and every later token is decoded by the batched step.
//!
//! Both engines are built with the same quantized KV mode and the same context,
//! so the quantized caches have the same head-major row layout: a slot is a
//! byte-for-byte copy, not a requantization.

use super::*;

impl<'a> BatchEngine<'a> {
    /// Free slots left in the pool.
    pub fn available_slots(&self) -> usize {
        self.slots - self.active.len()
    }

    /// Active slot indices, in the order `forward` consumes tokens.
    pub fn active_slots(&self) -> &[usize] {
        &self.active
    }

    /// Adopt the state currently in `src` as a new batch slot at `pos`.
    ///
    /// Returns the slot index. The caller must next feed the last emitted token
    /// (the one the single-sequence state has not appended yet) through
    /// `forward`, exactly as it would for a decode step.
    pub fn adopt(&mut self, src: &GpuEngine<'a>, pos: usize) -> Result<usize, String> {
        if src.layers.len() != self.layers.len() {
            return Err(format!(
                "adopt: source has {} layers, batch has {}",
                src.layers.len(),
                self.layers.len()
            ));
        }
        let slot = self.pool.alloc().ok_or("adopt: slot pool exhausted")?;
        let l = self.model.launcher();
        let stream = l.stream();
        let result = (|| -> Result<(), String> {
            for (sb, db) in src.layers.iter().zip(self.layers.iter()) {
                match (&sb.kind, &db.kind) {
                    (BlockKind::Full { deep: Some(dk), .. }, BatchKind::Full(f)) => {
                        if dk.dtype != f.kv.dtype {
                            return Err("adopt: KV dtype differs between engines".to_string());
                        }
                        let row = dk.row_words();
                        let heads = dk.heads();
                        let per = pos * row;
                        let slot_words = f.kv.slot_bytes() / 4;
                        let head_stride = dk.seq * row;
                        if (heads * head_stride) > slot_words {
                            return Err("adopt: KV cache does not fit the slot".to_string());
                        }
                        let base = slot * slot_words;
                        for h in 0..heads {
                            let src = h * head_stride;
                            let dst = base + h * head_stride;
                            dk.k.copy_range_on(&f.kv.k, src, dst, per, stream)?;
                            dk.v.copy_range_on(&f.kv.v, src, dst, per, stream)?;
                        }
                    }
                    (BlockKind::Ssm(s), BatchKind::Ssm(d)) => {
                        let (conv_len, state_len) = s.state_lens();
                        let cw = d.conv_state.len() / self.slots;
                        let sw = d.s_state.len() / self.slots;
                        if conv_len > cw || state_len > sw {
                            return Err("adopt: SSM state does not fit the slot".to_string());
                        }
                        s.conv_state_buf().copy_to_offset_on(
                            &d.conv_state,
                            slot * cw,
                            conv_len,
                            stream,
                        )?;
                        s.s_state_buf().copy_to_offset_on(
                            &d.s_state,
                            slot * sw,
                            state_len,
                            stream,
                        )?;
                    }
                    _ => return Err("adopt: layer kinds differ between engines".to_string()),
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.pool.free(slot);
            return Err(e);
        }
        l.synchronize()?;
        self.positions[slot] = pos;
        self.active.push(slot);
        Ok(slot)
    }

    /// Retire a slot. Its buffers stay allocated for the next request; the next
    /// `adopt` overwrites every word it reads.
    pub fn release(&mut self, slot: usize) -> Result<(), String> {
        let Some(i) = self.active.iter().position(|&s| s == slot) else {
            return Err(format!("release: slot {slot} is not active"));
        };
        self.active.remove(i);
        self.positions[slot] = 0;
        self.pool.free(slot);
        Ok(())
    }

    /// The engine's per-slot KV and SSM bytes, for a VRAM estimate.
    pub fn state_bytes(&self) -> usize {
        self.kv_bytes() + self.ssm_state_bytes()
    }
}
