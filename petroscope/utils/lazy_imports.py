from petroscope.utils import logger
import importlib


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

    def _load_module(self):
        if self._module is None:
            try:
                logger.info(f"Loading module: {self.module_name}")
                self._module = importlib.import_module(self.module_name)
            except ImportError:
                logger.error(self.error_message)
                exit(1)
        return self._module

    def __getattr__(self, name):
        module = self._load_module()
        try:
            return getattr(module, name)
        except AttributeError:
            # If the attribute is not found, assume it's a submodule and lazily import it
            submodule_name = f"{self.module_name}.{name}"
            return LazyImport(submodule_name, self.error_message)

    def __call__(self, *args, **kwargs):
        module = self._load_module()
        return module(*args, **kwargs)


_msg_torch = (
    "PyTorch is required to use this module. Please install it "
    "by following the instructions at "
    "https://pytorch.org/get-started/locally/."
)
_msg_torchvision = (
    "Torchvision is required to use this module. Please install it "
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
F = LazyImport("torch.nn.functional", error_message=_msg_torch)
model_zoo = LazyImport("torch.utils.model_zoo", error_message=_msg_torch)
models = LazyImport("torchvision.models", error_message=_msg_torchvision)
kornia = LazyImport("kornia", error_message=_msg_kornia)
