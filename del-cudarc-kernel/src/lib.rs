pub const ARRAY1D: &str = include_str!(concat!(env!("OUT_DIR"), "/array1d.ptx"));
pub const CUMSUM: &str = include_str!(concat!(env!("OUT_DIR"), "/cumsum.ptx"));
pub const GET_FLAGGED_ELEMENT: &str =
    include_str!(concat!(env!("OUT_DIR"), "/get_flagged_element.ptx"));
pub const OFFSET_ARRAY: &str = include_str!(concat!(env!("OUT_DIR"), "/offset_array.ptx"));
pub const SIMPLE: &str = include_str!(concat!(env!("OUT_DIR"), "/simple.ptx"));
pub const SORT_BY_KEY_U32: &str = include_str!(concat!(env!("OUT_DIR"), "/sort_by_key_u32.ptx"));
pub const SORT_BY_KEY_U64: &str = include_str!(concat!(env!("OUT_DIR"), "/sort_by_key_u64.ptx"));
pub const SORT_U32: &str = include_str!(concat!(env!("OUT_DIR"), "/sort_u32.ptx"));
pub const SORT_U64: &str = include_str!(concat!(env!("OUT_DIR"), "/sort_u64.ptx"));
pub const SORTED_ARRAY1D: &str = include_str!(concat!(env!("OUT_DIR"), "/sorted_array1d.ptx"));
pub const UTIL: &str = include_str!(concat!(env!("OUT_DIR"), "/util.ptx"));
