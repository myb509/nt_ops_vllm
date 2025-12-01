import importlib
import nt_ops
import logging
logger = logging.getLogger(__name__)
# A list of tuples defining the patches to apply.
# Each tuple contains:
# (module_path, object_name, attribute_to_patch, patch_function)
# If object_name is None, the attribute is patched directly on the module.
_PATCHES = [
    ('vllm.model_executor.layers.utils', None, 'dispatch_unquantized_gemm',
     nt_ops.linear.dispatch_unquantized_gemm),
    ('vllm.model_executor.layers.activation', 'SiluAndMul', 'forward',
     nt_ops.activation.silu_and_mul_forward),
    ('vllm.model_executor.layers.layernorm', 'RMSNorm', 'forward',
     nt_ops.rms.rms_forward),
]


def apply_monkey_patches():
    """
    Applies all monkey patches defined in the _PATCHES list.
    This function is executed when the module is imported.
    """
    for module_path, obj_name, attr_name, patch_func in _PATCHES:
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the target object to patch
            target_obj = getattr(module, obj_name) if obj_name else module

            # Apply the patch
            setattr(target_obj, attr_name, patch_func)

            target_name = f"{module_path}.{obj_name}" if obj_name else module_path
            logger.warning(f"\033[31mSuccessfully patched {target_name}.{attr_name}.\033[0m")

        except (ImportError, AttributeError) as e:
            logger.warning(f"\033[31mFailed to apply patch for {module_path}: {e}\033[0m")


# Apply all patches when this module is imported.
apply_monkey_patches()