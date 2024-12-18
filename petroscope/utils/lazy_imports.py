from loguru import logger


class LazyImport:
    def __init__(self, module_name, error_message=None, immediate_load=False):
        self.module_name = module_name
        self.error_message = (
            error_message
            or f"Module '{module_name}' is required but not installed."
        )
        self._module = None
        if immediate_load:
            self._load_module()

    def _is_module_installed(self):
        try:
            __import__(self.module_name)
            return True
        except ImportError:
            return False

    def _load_module(self):
        if self._module is None:
            try:
                logger.info(f"loading module: {self.module_name}")
                self._module = __import__(self.module_name)
            except ImportError:
                logger.error(self.error_message)
                exit(1)
                raise ImportError(self.error_message) from None
        return self._module

    def __getattr__(self, name):
        module = self._load_module()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load_module()
        return module(*args, **kwargs)


_msg_torch = (
    "PyTorch is required to use this module. Please install it "
    "by following the instructions at "
    "https://pytorch.org/get-started/locally/."
)
_msg_kornia = (
    "Kornia is required to use this module. Please install it "
    "by following the instructions at "
    "https://kornia.readthedocs.io/en/latest/get-started/installation.html"
)


torch = LazyImport("torch", error_message=_msg_torch)
nn = LazyImport("torch.nn", error_message=_msg_torch)
optim = LazyImport("torch.optim", error_message=_msg_torch)
kornia = LazyImport("kornia", error_message=_msg_kornia)
