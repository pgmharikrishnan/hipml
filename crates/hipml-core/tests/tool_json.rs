//! Tool-argument formatting tests (nlohmann-compatible numbers, escaping and
//! the template's `<tool_call>` spacing).

#[cfg(test)]
mod tests {
    use hipml_core::tool_json::*;

    #[test]
    fn numbers_match_nlohmann_dump() {
        for (src, want) in [
            ("1", "1"),
            ("-2", "-2"),
            ("1.0", "1.0"),
            ("1e3", "1000.0"),
            ("0.1", "0.1"),
            ("1e-7", "1e-07"),
            ("1.5", "1.5"),
        ] {
            let v = parse(src).unwrap();
            assert_eq!(tojson(&v), want, "{src}");
        }
        let v = parse(r#"{"a":1,"b":[true,null,"x"],"c":{"k":"v"}}"#).unwrap();
        assert_eq!(
            tojson(&v),
            r#"{"a": 1, "b": [true, null, "x"], "c": {"k": "v"}}"#
        );
    }

    #[test]
    fn strings_in_objects_are_escaped_but_top_level_values_are_raw() {
        let v = parse(r#"{"s":["a\tb\u0001"],"plain":"a\tb"}"#).unwrap();
        assert_eq!(
            tojson(&v),
            "{\"s\": [\"a\\tb\\u0001\"], \"plain\": \"a\\tb\"}"
        );
    }

    #[test]
    fn render_matches_template_spacing() {
        let calls = vec![
            ToolCall {
                name: "get_weather".into(),
                arguments: vec![
                    ("city".into(), Arg::Str("Paris".into())),
                    ("n".into(), Arg::Int(1)),
                ],
            },
            ToolCall {
                name: "done".into(),
                arguments: Vec::new(),
            },
        ];
        let mut out = String::new();
        render_calls(&mut out, "checking", &calls);
        assert_eq!(
            out,
            "\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
             </parameter>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=done>\n</function>\n</tool_call>"
        );
    }
}
