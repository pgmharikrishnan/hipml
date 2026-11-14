// SPDX-License-Identifier: Apache-2.0
//! Chat-template rendering tests: tool calls, thinking preservation, content
//! parts. Moved out of `chat.rs` to keep that module under the file-size cap.

#[cfg(test)]
mod tests {
    use hipml_core::chat::*;
    use hipml_core::tool_json::{Arg, ToolCall};

    const TEMPLATE: &str = "{%- macro render_content(content, do_vision_count) %}\
        {%- if content is string %}{{- content }}{%- endif %}{%- endmacro %}\
        {%- set reasoning_instructions = '' %}\
        {%- if preserve_thinking is undefined or preserve_thinking is true %}{%- endif %}\
        {%- if enable_thinking is undefined or enable_thinking is true %}{%- endif %}\
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant' }}{%- endif %}\
        {{- '<|im_end|>' }}{{- '<think>' }}";

    #[test]
    fn single_turn_thinking() {
        let msgs = [Message::new("user", "hi")];
        let out = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap();
        assert!(out.starts_with("<|im_start|>system\nReasoning effort is set to xhigh."));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>\n"));
    }

    #[test]
    fn thinking_off_has_empty_block() {
        let opts = ChatOptions {
            enable_thinking: false,
            ..ChatOptions::default()
        };
        let out = render(TEMPLATE, &[Message::new("user", "hi")], &opts).unwrap();
        assert!(!out.contains("Reasoning effort"));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn rejects_unknown_template() {
        let err = render(
            "plain chatml",
            &[Message::new("user", "hi")],
            &ChatOptions::default(),
        );
        assert!(matches!(err, Err(ChatError::Unsupported(_))));
    }

    #[test]
    fn preserve_thinking_false_drops_the_pre_query_turn() {
        let msgs = [
            Message::new("user", "q1"),
            Message::assistant_with_reasoning("a1", "secret"),
            Message::new("user", "q2"),
        ];
        let opts = ChatOptions {
            preserve_thinking: false,
            ..ChatOptions::default()
        };
        let out = render(TEMPLATE, &msgs, &opts).unwrap();
        assert!(out.contains("<|im_start|>assistant\na1<|im_end|>\n"));
        assert!(!out.contains("secret"));
        let kept = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap();
        assert!(kept.contains("<think>\nsecret\n</think>\n\na1"));
    }

    #[test]
    fn preserve_thinking_false_keeps_the_post_query_turn() {
        // `loop.index0 > last_query_index` in the template: an assistant turn
        // after the last real user query keeps its <think> block.
        let msgs = [
            Message::new("user", "q"),
            Message::assistant_with_reasoning("a", "why"),
        ];
        let opts = ChatOptions {
            preserve_thinking: false,
            ..ChatOptions::default()
        };
        let out = render(TEMPLATE, &msgs, &opts).unwrap();
        assert!(out.contains("<|im_start|>assistant\n<think>\nwhy\n</think>\n\na"));
    }

    #[test]
    fn tool_calls_follow_content() {
        let call = ToolCall {
            name: "get_weather".into(),
            arguments: vec![("city".into(), Arg::Str("Paris".into()))],
        };
        let msgs = [
            Message::new("user", "hi"),
            Message::new("assistant", "checking").with_tool_calls(vec![call]),
            Message::new("tool", "18C"),
        ];
        let out = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap();
        assert!(out.contains(
            "<|im_start|>assistant\n<think>\n\n</think>\n\nchecking\n\n<tool_call>\n\
             <function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
             </function>\n</tool_call><|im_end|>\n"
        ));
    }

    #[test]
    fn image_and_video_parts_are_rejected() {
        for part in [ContentPart::Image, ContentPart::Video] {
            let msgs = [Message::new("user", "").with_parts(vec![part])];
            let err = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap_err();
            assert!(matches!(err, ChatError::Unsupported(ref m) if m.contains("not supported")));
        }
    }

    #[test]
    fn text_parts_concatenate_without_a_separator() {
        let msgs = [Message::new("user", "").with_parts(vec![
            ContentPart::Text("hello ".into()),
            ContentPart::Text("world".into()),
        ])];
        let out = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap();
        assert!(out.contains("<|im_start|>user\nhello world<|im_end|>\n"));
    }

    /// MiniCPM-V-4.7's template: ChatML turns and a thinking tag, no
    /// `preserve_thinking`, no `reasoning_instructions`, and thinking hard
    /// disabled at the generation prompt.
    const MINICPM: &str = "{%- macro render_content(content) %}{{- content -}}{%- endmacro %}\
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant' }}{%- endif %}\
        {{- '<|im_end|>' }}{{- '<think>' }}";

    #[test]
    fn template_without_preserve_thinking_renders() {
        // The 400 this fixes: `preserve_thinking` was a hard-required marker,
        // so every MiniCPM-V-4.7 request failed before the model was reached.
        let msgs = [
            Message::new("user", "q1"),
            Message::assistant_with_reasoning("a1", "secret"),
            Message::new("user", "q2"),
        ];
        let out = render(MINICPM, &msgs, &ChatOptions::default()).unwrap();
        assert!(out.contains("<|im_start|>user\nq1<|im_end|>\n"));
        // No branch keeps a pre-query thinking block, so none is emitted.
        assert!(!out.contains("secret"));
        assert!(out.contains("<|im_start|>assistant\na1<|im_end|>\n"));
    }

    #[test]
    fn template_with_unknown_marker_renders() {
        // A marker the engine has never seen must not fail a render: the
        // validator only inspects the two ChatML delimiters.
        let template = MINICPM.replace(
            "{%- macro render_content",
            "{%- if some_future_flag %}{%- endif %}{%- macro render_content",
        );
        let out = render(
            &template,
            &[Message::new("user", "hi")],
            &ChatOptions::default(),
        )
        .unwrap();
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>\n"));
    }

    #[test]
    fn template_without_enable_thinking_cannot_open_a_block() {
        // `enable_thinking` is the template's knob. MiniCPM's template does not
        // read it and hard-codes the closed block, so asking for thinking must
        // still render the closed one.
        let out = render(
            MINICPM,
            &[Message::new("user", "hi")],
            &ChatOptions::default(),
        )
        .unwrap();
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let opts = ChatOptions {
            enable_thinking: false,
            ..ChatOptions::default()
        };
        let off = render(MINICPM, &[Message::new("user", "hi")], &opts).unwrap();
        assert_eq!(out, off);
    }

    #[test]
    fn tojson_matches_llama_spacing() {
        assert_eq!(
            tojson(r#"{"type":"function","n":[1,2]}"#),
            r#"{"type": "function", "n": [1, 2]}"#
        );
        assert_eq!(
            tojson(r#"{"s":"a, b: c","e":"x\"y"}"#),
            r#"{"s": "a, b: c", "e": "x\"y"}"#
        );
    }
}
