use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    token::{Colon, Comma, DotDot},
    LitInt, Token,
};

/// 切片参数类型
enum SliceArg {
    Index(usize),
    Range(usize, usize, usize), // start, end, step
    All,
}

impl Parse for SliceArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // 处理 `..`
        if input.peek(DotDot) {
            input.parse::<DotDot>()?;
            return Ok(SliceArg::All);
        }

        // 解析起始整数
        let start_lit: LitInt = input.parse()?;
        let start = start_lit.base10_parse::<usize>()?;

        // 检查是否有 `..`
        if input.peek(DotDot) {
            input.parse::<DotDot>()?;
            // 解析结束整数（可选）
            let end = if input.is_empty() || input.peek(Comma) || input.peek(Colon) {
                usize::MAX
            } else {
                let end_lit: LitInt = input.parse()?;
                end_lit.base10_parse::<usize>()?
            };
            // 检查是否有 `:`
            let step = if input.peek(Colon) {
                input.parse::<Colon>()?;
                let step_lit: LitInt = input.parse()?;
                step_lit.base10_parse::<usize>()?
            } else {
                1
            };
            Ok(SliceArg::Range(start, end, step))
        } else {
            Ok(SliceArg::Index(start))
        }
    }
}

struct SliceInfo {
    args: Vec<SliceArg>,
}

impl Parse for SliceInfo {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = Vec::new();
        while !input.is_empty() {
            args.push(input.parse::<SliceArg>()?);
            if !input.peek(Comma) {
                break;
            }
            input.parse::<Comma>()?;
        }
        Ok(SliceInfo { args })
    }
}

#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    let info = parse_macro_input!(input as SliceInfo);
    let args = info.args.into_iter().map(|arg| match arg {
        SliceArg::Index(idx) => quote! { crate::view::SliceArg::Index(#idx) },
        SliceArg::Range(start, end, step) => {
            quote! { crate::view::SliceArg::Range(#start, #end, #step) }
        }
        SliceArg::All => quote! { crate::view::SliceArg::All },
    });
    let expanded = quote! {
        crate::view::SliceInfo::new(vec![ #(#args),* ])
    };
    expanded.into()
}
