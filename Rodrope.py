from types import MethodType
from functools import partial
import Rodrope_patch as RP

def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the method of an instance of a model class. 
        It's part from chat-GPT.
        It will replace the method  with the new method.
        Currently, we only use this function to modify the attention method of a model. Do not test it further. 

        instance: 
            instance of a model to modify.
        target_class_name: 
            name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original method. E.g. 'rodrope_forward'. 
            It should include a parameter 'self' to be binded to the instance.
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        target_found = False
        return target_found
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found


def apply(
    loaded_model,
    group_size=None,
    window_size=None,
    enable_flash_attention=False,
    scale_base=-1,
    flash_attention_impl="triton",
    far_size=None,
    far_group_size=None,
    far2_size=None,
    far2_group_size=None,
    block=2,
    m1=None,
    lambda2=None,
    m2=None,
    lambda3=None,
    m3=None,
    lambda4=None,
    rodrope_block=None,
):
    '''
        loaded_model:
            model to apply the self-attention extension.
        m1 / window_size:
            first boundary; the initial local window keeps original RoPE.
        lambda2 / group_size:
            code compression factor for the second block.
        m2 / far_size:
            optional boundary where the third/far block starts. Required when block=3/4.
        lambda3 / far_group_size:
            optional code compression factor for the third/far block. Required when block=3/4.
        m3 / far2_size:
            optional boundary where the fourth/far2 block starts. Required when block=4.
        lambda4 / far2_group_size:
            optional code compression factor for the fourth/far2 block. Required when block=4.
        rodrope_block / block:
            2 for original Rodrope; 3 for neighbor/group/far three-block Rodrope;
            4 for neighbor/group/far/far2 four-block Rodrope.

        Backward compatibility:
            Older code can still pass group_size/window_size/far_size/far_group_size.
            New experiment scripts can pass m1/lambda2/m2/lambda3/m3/lambda4.
        scale_base:
            base for the scale, equal to pretraining length.
            e.g. 4096 for Llama, 8192 for Gemma

            Two recommended scale factor:
                yarn: https://arxiv.org/abs/2309.00071
                log: https://arxiv.org/abs/2202.12172 ; https://kexue.fm/archives/8823
            This is helpful while retrieving a long sequence (e.g a long passkey).
            But on real-world data, the impact is minor. (e.g. on LongBench, LEval).

            The reported results in our paper does not use this scale except for long passkey retrieval.
    '''
    def resolve_alias(primary_name, primary_value, alias_name, alias_value):
        if alias_value is None:
            return primary_value
        if primary_value is not None and primary_value != alias_value:
            raise ValueError(f"Both {primary_name} and {alias_name} were provided with different values.")
        return alias_value

    block = resolve_alias("block", block, "rodrope_block", rodrope_block)
    window_size = resolve_alias("window_size", window_size, "m1", m1)
    group_size = resolve_alias("group_size", group_size, "lambda2", lambda2)
    far_size = resolve_alias("far_size", far_size, "m2", m2)
    far_group_size = resolve_alias("far_group_size", far_group_size, "lambda3", lambda3)
    far2_size = resolve_alias("far2_size", far2_size, "m3", m3)
    far2_group_size = resolve_alias("far2_group_size", far2_group_size, "lambda4", lambda4)

    if group_size is None or window_size is None:
        raise ValueError("Rodrope.apply requires lambda2/m1 (or legacy group_size/window_size).")
    block = int(block)

    if block not in (2, 3, 4):
        raise ValueError("block should be 2 for original Rodrope, 3 for three-block Rodrope or 4 for four-block Rodrope.")
    if block in (3, 4) and not enable_flash_attention:
        raise NotImplementedError("Three/four-block Rodrope is implemented for flash attention only.")
    if block == 2:
        far_size = None
        far_group_size = None
        far2_size = None
        far2_group_size = None
    elif far_size is None or far_group_size is None:
        raise ValueError(f"block={block} requires far_size and far_group_size.")
    elif block == 4 and (far2_size is None or far2_group_size is None):
        raise ValueError("block=4 requires far2_size and far2_group_size.")

    arch_name = loaded_model.__class__.__name__
    if block == 3 and not ('Llama' in arch_name or 'Mistral' in arch_name):
        raise NotImplementedError("Three-block Rodrope is currently implemented for Llama/Mistral flash_attn only.")
    if block == 4 and 'Llama' not in arch_name:
        raise NotImplementedError("Four-block Rodrope is currently implemented for Llama flash_attn only.")
    if 'Llama' in arch_name:
        if enable_flash_attention:
            if flash_attention_impl == "flash_attn":
                if block == 2:
                    rodrope_attention_forward = partial(RP.Llama.flash_rodrope_two_block_forward,
                                                lambda2=group_size,
                                                m1=window_size,
                                                scale_base=scale_base)
                elif block == 3:
                    rodrope_attention_forward = partial(RP.Llama.flash_rodrope_three_block_forward,
                                                lambda2=group_size,
                                                m1=window_size,
                                                m2=far_size,
                                                lambda3=far_group_size,
                                                block=block,
                                                scale_base=scale_base)
                else:
                    rodrope_attention_forward = partial(RP.Llama.flash_rodrope_four_block_forward,
                                                lambda2=group_size,
                                                m1=window_size,
                                                m2=far_size,
                                                lambda3=far_group_size,
                                                m3=far2_size,
                                                lambda4=far2_group_size,
                                                block=block,
                                                scale_base=scale_base)
                modifed_1 = modify_method_of_instance(loaded_model, "LlamaFlashAttention2", "_flash_attention_forward", RP.Rodrope_flash_attn.flash_attention2_forward_with_window_size)
                modifed_2 = modify_method_of_instance(loaded_model, "LlamaFlashAttention2", "forward", rodrope_attention_forward)
                print(f"Using flash_attn flash rodrope block={block}!!")
                if (not modifed_1) or (not modifed_2):
                    raise Exception(f"Failed to modify the attention method of {arch_name}")

            elif flash_attention_impl == "triton":
                if block in (3, 4):
                    raise NotImplementedError("Three/four-block Rodrope is implemented for flash_attention_impl='flash_attn'.")
                rodrope_attention_forward = partial(RP.Llama.flash_rodrope_forward_triton,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
                modifed = modify_method_of_instance(loaded_model, "LlamaFlashAttention2", "forward", rodrope_attention_forward)
                print("Using triton flash rodrope!!")
                if (not modifed):
                    raise Exception(f"Failed to modify the attention method of {arch_name}")
            else:
                raise Exception(f"Need to set the flash_attention_impl to 'flash_attn' or 'triton'.")


        else:
            rodrope_attention_forward = partial(RP.Llama.rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            # after the default version of attention in 4.36 is LlamaSpdaAttention, but in before 4,36 or in 4.38, it is LlamaAttention
            # print("loaded_model", loaded_model)
            modifed_2 = modify_method_of_instance(loaded_model, "LlamaAttention", "forward", rodrope_attention_forward)
            # if not modifed_2:
            #     modifed_2 = modify_method_of_instance(
            #         loaded_model, "LlamaSdpaAttention", "forward", rodrope_attention_forward
            #     )
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Mistral' in arch_name:
        # Mistral shares the Llama-style Rodrope implementation, including 3block in flash_attn mode.
        if enable_flash_attention:
            if flash_attention_impl != "flash_attn":
                if block in (3, 4):
                    raise NotImplementedError("Mistral three/four-block Rodrope requires flash_attention_impl='flash_attn'.")
                raise Exception(f"Need to set the flash_attention_impl to 'flash_attn'.")
            if block == 2:
                rodrope_attention_forward = partial(RP.Mistral.flash_rodrope_two_block_forward,
                                                group_size_1=group_size,
                                                group_size_2=window_size,
                                                scale_base=scale_base)
            else:
                rodrope_attention_forward = partial(RP.Mistral.flash_rodrope_three_block_forward,
                                                group_size_1=group_size,
                                                group_size_2=window_size,
                                                far_size=far_size,
                                                far_group_size=far_group_size,
                                                block=block,
                                                scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "MistralFlashAttention2", "_flash_attention_forward", RP.Rodrope_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "MistralFlashAttention2", "forward", rodrope_attention_forward)
            print(f"Using Mistral flash_attn rodrope block={block}!!")
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            if block in (3, 4):
                raise NotImplementedError("Mistral three/four-block Rodrope is implemented for flash attention only.")
            rodrope_attention_forward = partial(RP.Mistral.rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "MistralAttention", "forward", rodrope_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Gemma' in arch_name:
        if enable_flash_attention:
            rodrope_attention_forward = partial(RP.Gemma.flash_rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "GemmaFlashAttention2", "_flash_attention_forward", RP.Rodrope_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "GemmaFlashAttention2", "forward", rodrope_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            rodrope_attention_forward = partial(RP.Gemma.rodrope_forward,
                                            group_size_1=group_size,
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2= modify_method_of_instance(loaded_model, "GemmaAttention", "forward", rodrope_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Qwen2' in arch_name:
        if enable_flash_attention:
            rodrope_attention_forward = partial(RP.Qwen2.flash_rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "_flash_attention_forward", RP.Rodrope_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "forward", rodrope_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            rodrope_attention_forward = partial(RP.Qwen2.rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "Qwen2Attention", "forward", rodrope_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    elif 'Phi' in arch_name:
        if enable_flash_attention:
            rodrope_attention_forward = partial(RP.Phi.flash_rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_1 = modify_method_of_instance(loaded_model, "PhiFlashAttention2", "_flash_attention_forward", RP.Rodrope_flash_attn.flash_attention2_forward_with_window_size)
            modifed_2 = modify_method_of_instance(loaded_model, "PhiFlashAttention2", "forward", rodrope_attention_forward)
            if (not modifed_1) or (not modifed_2):
                raise Exception(f"Failed to modify the attention method of {arch_name}")
        else:
            rodrope_attention_forward = partial(RP.Phi.rodrope_forward,
                                            group_size_1=group_size, 
                                            group_size_2=window_size,
                                            scale_base=scale_base)
            modifed_2 = modify_method_of_instance(loaded_model, "PhiAttention", "forward", rodrope_attention_forward)
            if not modifed_2:
                raise Exception(f"Failed to modify the attention method of {arch_name}")
    else:
        raise NotImplementedError

