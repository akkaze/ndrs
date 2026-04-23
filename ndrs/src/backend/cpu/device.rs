//! CPU 设备管理（单例）

use std::cell::RefCell;

thread_local! {
    static IS_CPU_SET: RefCell<bool> = const { RefCell::new(false) };
}

/// 设置 CPU 为当前设备（无参数，总是成功）
pub fn set_device() {
    IS_CPU_SET.with(|s| *s.borrow_mut() = true);
}

/// 获取当前是否为 CPU 设备
pub fn get_device() -> bool {
    IS_CPU_SET.with(|s| *s.borrow())
}
