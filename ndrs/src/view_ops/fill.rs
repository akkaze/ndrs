#[macro_export]
macro_rules! impl_fill {
    ($view_type:ident, $handle:ty) => {
        fn fill<T: bytemuck::Pod + crate::dtype::DTypeMapping>(&mut self, value: T) -> Result<()> {
            match self.device() {
                Device::Cpu => {
                    // 调用 CPU 填充实现（假设有 Self::fill_cpu 静态方法）
                    Self::fill_cpu(self, value)
                }
                Device::Cuda(_) => {
                    // 调用 GPU 填充实现（假设有 Self::fill_cuda 静态方法）
                    Self::fill_cuda(self, value)
                }
            }
        }
    };
}
