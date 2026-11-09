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
        let slot = self.active.len();
        if slot >= self.slots {
            return Err("adopt: slot pool exhausted".to_string());
        }
        self.copy_in(src, slot, pos)?;
        debug_assert_eq!(slot, self.active.len());
        self.active.push(slot);
        Ok(slot)
    }

    /// Copy a prefilled state into `slot` without activating it.
    ///
    /// The row-block prefill uses this: several quiet slots are filled from
    /// their prefix state, one `prefill_rows` call advances them, and only then
    /// are they activated in the same index order, keeping the active set
    /// dense. Quiet slots are the range `active.len()..` in gather order.
    pub fn adopt_quiet(
        &mut self,
        src: &GpuEngine<'a>,
        slot: usize,
        pos: usize,
    ) -> Result<(), String> {
        if slot < self.active.len() || slot >= self.slots {
            return Err(format!(
                "adopt_quiet: slot {slot} outside {}..{}",
                self.active.len(),
                self.slots
            ));
        }
        self.copy_in(src, slot, pos)
    }

    /// First slot a quiet adopt would use (the end of the dense active set).
    pub fn next_slot(&self) -> usize {
        self.active.len()
    }

    /// Activate the next quiet slot. `slot` must be `active.len()`.
    pub fn activate_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot != self.active.len() {
            return Err(format!(
                "activate_slot: slot {slot} is not the next free index {}",
                self.active.len()
            ));
        }
        if slot >= self.slots {
            return Err("activate_slot: slot pool exhausted".to_string());
        }
        self.active.push(slot);
        Ok(())
    }

    /// Copy `src`'s live state into `slot` at `pos`, leaving `active` alone.
    fn copy_in(&mut self, src: &GpuEngine<'a>, slot: usize, pos: usize) -> Result<(), String> {
        if src.layers.len() != self.layers.len() {
            return Err(format!(
                "adopt: source has {} layers, batch has {}",
                src.layers.len(),
                self.layers.len()
            ));
        }
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
            return Err(e);
        }
        l.synchronize()?;
        self.positions[slot] = pos;
        Ok(())
    }

    /// Retire a slot, keeping the active set dense.
    ///
    /// The kernels address per-slot state by `grid.y`, so the last active slot
    /// is copied down into the freed index. Returns the old index that moved
    /// (always greater than `slot`), so the caller can move its own bookkeeping
    /// too. Buffers stay allocated; the next `adopt` overwrites every word it
    /// reads at the new tail.
    pub fn release(&mut self, slot: usize) -> Result<Option<usize>, String> {
        let Some(i) = self.active.iter().position(|&s| s == slot) else {
            return Err(format!("release: slot {slot} is not active"));
        };
        if i != slot {
            return Err(format!("release: active[{i}] is slot {slot}, want a dense set"));
        }
        let last = self.active.len() - 1;
        if slot == last {
            self.active.pop();
            self.positions[slot] = 0;
            return Ok(None);
        }
        // The active set is dense (0..n), so popping the tail leaves `slot`
        // in place; only the device state has to move down.
        self.copy_slot(last, slot)?;
        self.positions[slot] = self.positions[last];
        self.positions[last] = 0;
        self.active.pop();
        self.model.launcher().synchronize()?;
        Ok(Some(last))
    }

    /// Device-to-device copy of one slot's live KV rows and SSM state.
    fn copy_slot(&self, from: usize, to: usize) -> Result<(), String> {
        let l = self.model.launcher();
        let stream = l.stream();
        let pos = self.positions[from];
        for b in &self.layers {
            match &b.kind {
                BatchKind::Full(f) => {
                    let bytes = f.kv.row();
                    if bytes % 4 != 0 {
                        return Err(format!("release: KV row {bytes} B is not word aligned"));
                    }
                    let row = bytes / 4;
                    let slot_words = f.kv.slot_bytes() / 4;
                    let head_stride = f.kv.seq * row;
                    let per = pos * row;
                    let src = from * slot_words;
                    let dst = to * slot_words;
                    for h in 0..f.kv.n_kv {
                        let o = h * head_stride;
                        f.kv.k.copy_range_on(&f.kv.k, src + o, dst + o, per, stream)?;
                        f.kv.v.copy_range_on(&f.kv.v, src + o, dst + o, per, stream)?;
                    }
                }
                BatchKind::Ssm(d) => {
                    let (conv_len, state_len) = d.inst.state_lens();
                    let cw = d.conv_state.len() / self.slots;
                    let sw = d.s_state.len() / self.slots;
                    d.conv_state.copy_range_on(
                        &d.conv_state,
                        from * cw,
                        to * cw,
                        conv_len,
                        stream,
                    )?;
                    d.s_state.copy_range_on(&d.s_state, from * sw, to * sw, state_len, stream)?;
                }
            }
        }
        Ok(())
    }

    /// The engine's per-slot KV and SSM bytes, for a VRAM estimate.
    pub fn state_bytes(&self) -> usize {
        self.kv_bytes() + self.ssm_state_bytes()
    }
}
