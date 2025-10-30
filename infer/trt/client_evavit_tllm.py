import os
import sys
import json
import argparse
import numpy as np
import time
import base64
import torch
from functools import partial
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from LLMTextHandler.tritonft.utils import prepare_tensor
from LLMTextHandler.handler_manager import get_handler

from .modality_utils import prepare_trt_inputs, has_output_diamond_question_mark
from .modality_utils import parse_list_inputs
from .modality_utils import get_vit_infos, grpc_infer_vit

# NOTE: special token map
special_token_map = {
    "<p>": "<|extra_61|>",
    "</p>": "<|extra_62|>",
    "<box>": "<|extra_63|>",
    "</box>": "<|extra_64|>",
    "<quad>": "<|extra_66|>",
    "</quad>": "<|extra_67|>",
    "<ref>": "<|extra_68|>",
    "</ref>": "<|extra_69|>",
    "<pFig>": "<|extra_94|>",
    "</pFig>": "<|extra_95|>",
    "<det>": "<|extra_101|>",
    "</det>": "<|extra_102|>",
    "<point>": "<|extra_103|>",
    "</point>": "<|extra_104|>"
}

new_special_token_map = {
    "<p>": "<｜hy_place▁holder▁no▁106｜>",
    "</p>": "<｜hy_place▁holder▁no▁107｜>",
    "<box>": "<｜hy_place▁holder▁no▁108｜>",
    "</box>": "<｜hy_place▁holder▁no▁109｜>",
    "<quad>": "<｜hy_place▁holder▁no▁110｜>",
    "</quad>": "<｜hy_place▁holder▁no▁111｜>",
    "<ref>": "<｜hy_place▁holder▁no▁112｜>",
    "</ref>": "<｜hy_place▁holder▁no▁113｜>",
    "<pFig>": "<｜hy_place▁holder▁no▁114｜>",
    "</pFig>": "<｜hy_place▁holder▁no▁115｜>",
    "<det>": "<｜hy_place▁holder▁no▁116｜>",
    "</det>": "<｜hy_place▁holder▁no▁117｜>",
    "<point>": "<｜hy_place▁holder▁no▁118｜>",
    "</point>": "<｜hy_place▁holder▁no▁119｜>"
}

special_token_map_inv = {v: k for k, v in special_token_map.items()}
new_special_token_map_inv = {v: k for k, v in new_special_token_map.items()}
def replace_special_tokens(prompt, replace_dict):
    origin_text = prompt
    for ori_text_token, special_text_token in replace_dict.items():
        prompt = prompt.replace(ori_text_token, special_text_token)
    return prompt

def grpc_stream_infer(url, model_name, input_infos,
        img_emb_list, patch_width_list=[],
        tokenizer_type = 'hy_ptm_for_business_gqa',
        max_input_seq_len=16384, output_seq_len=2048,
        beam_size=1, top_k=1, top_p=1.0,
        random_seed=1234, diversity_rate=0.0,
        temperature=1.0, len_penalty=0.0, repetition_penalty=1.0,
        min_length=0, request_id="", is_stop=False,
        bad_words_list=list(), stop_words_list=list(), return_log_probs=False,
        return_top_log_probs=0, return_logits=False,
        return_hidden_states = False, return_last_context_embeddings = False,
        vocab_size=0, logit_bias=dict(),
        repetition_ngrams=None, repetition_patiences=None,
        opt_request_output_len=-1, opt_request_keep_iter=200, stream_mode=None,
        img_pos_reverse=True, verbose=False, exact_tokenizer=True,
        data_mode=0, xd_pos_num=0,
        valid_range_start_id=0, valid_range_end_id=0,
        special_id=-1, max_special_id_position=-1, system_prompt=None):

    if tokenizer_type == "hy_bos_v2_sft":
        token_map_inv = new_special_token_map_inv
        token_map = new_special_token_map
    else:
        token_map_inv = special_token_map_inv
        token_map = special_token_map
    # early stop
    if is_stop:
        assert len(str(request_id).strip()) > 0, "The stop request id should not be empty"
        if verbose:
            print(f"[WARNING]: The inputs will be ignored in early-stop mode", file=sys.stderr)
        with grpcclient.InferenceServerClient(url, verbose=False) as cl:
            inputs = [
                prepare_tensor(grpcclient, "input_ids", np.empty([1, 1]).astype(np.uint32)),
                prepare_tensor(grpcclient, "input_lengths", np.zeros([1, 1]).astype(np.uint32)),
                prepare_tensor(grpcclient, "request_output_len", np.zeros([1, 1]).astype(np.uint32)),
                prepare_tensor(grpcclient, "stop", is_stop * np.ones([1, 1]).astype(bool))
            ]

            cl.start_stream(callback=(
                lambda result, error: print(error, file=sys.stderr) if error else print(f"Request {request_id} stop success")
            ))
            cl.async_stream_infer(model_name, inputs, request_id=str(request_id))
        # nothing need to return in early stop mode
        return ""

    tokenizer = get_handler(tokenizer_type)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id

    if tokenizer_type == "hy_bos_v2_sft":
        img_start_id = 120118
        img_end_id = 120119
        img_token_id = 120120
        video_start_id = 120122
        video_end_id = 120123

    elif len(tokenizer.tokenizer.get_vocab()) == 127957:
        img_start_id = 127962
        img_end_id = 127963
        img_token_id = 127968
        video_start_id = 128001
        video_end_id = 128002

        # Noticed for modality
        eos_token_id = 127957
    elif len(tokenizer.tokenizer.get_vocab()) == 290943:
        img_start_id = 11
        img_end_id = 12
        img_token_id = 13
    else:
        raise NotImplementedError

    # NOTE: change eos_id to round_sep <|extra_9|>      127971
    # 
    if tokenizer_type == "hy_bos_v2_sft":
        bos_token_id = 120000
        system_prompt_id = 120021
        user_token_id = 120006
        sep_token_id = 120006
        eos_token_id = 120007
        eod_token_id = 120020
        # eos_token_id = 127971
        # eod_token_id = 127957
    else:
        # round_sep_id = 127971
        bos_token_id = tokenizer.tokenizer.encode(tokenizer.prompt_prefix)[0]
        system_prompt_id = 128062
        sep_token_id = 128000
        eos_token_id = 127971
        eod_token_id = 127957
        

    token_ids = []
    positions = []
    img_count = 0
    img_id_len = 0

    position_ids_t = []
    position_ids_x = []
    position_ids_y = []

    special_flag = False
    if special_flag:
        token_ids = [290943, 14915, 705, 10116, 1327, 767, 21039, 49792, 520, 290944, 705, 290943, 4508, 705]

    if exact_tokenizer:
        if tokenizer.prompt_prefix:
            # NOTE: add system_prompt_sep <|extra_100|> 128062
            # if tokenizer_type != "hy_bos_v2_sft":
            token_ids += [bos_token_id]
            if system_prompt:
                token_ids += tokenizer.tokenizer.encode(system_prompt)
            token_ids += [system_prompt_id]
            # else:
            #     token_ids += [bos_token_id]   # bos_token - <｜hy_begin▁of▁sentence｜>
            #     if system_prompt:
            #         token_ids += tokenizer.tokenizer.encode(system_prompt)
            #         token_ids += [system_prompt_id]   # system_prompt_sep - <｜hy_place▁holder▁no▁3｜>
            #     token_ids += [user_token_id]   # user_token - <｜hy_User｜>
            if xd_pos_num in [3, 4]:
                token_num = len(token_ids)
                position_ids_t += [token_num - 1]
                position_ids_x += [token_num - 1]
                position_ids_y += [token_num - 1]

    episode_flag = True
    for input_info in input_infos:
        if isinstance(input_info, dict):
            if input_info['image_path'] is not None:
                if data_mode in [1, 2, 3]:
                    if episode_flag:
                        token_ids += [video_start_id]
                        episode_flag = False
                        if xd_pos_num in [3, 4]:
                            token_num = len(token_ids)
                            position_ids_t += [token_num - 1]
                            position_ids_x += [token_num - 1]
                            position_ids_y += [token_num - 1]

                if special_flag:
                    img_tokens_num = img_emb_list[img_count].shape[2]
                    bias_pos = len(token_ids)
                    positions += list(range(bias_pos, bias_pos + img_tokens_num))
                    token_ids += [img_token_id] * img_tokens_num
                elif img_emb_list[img_count].shape[1] == 1:
                    img_tokens_num = img_emb_list[img_count].shape[2]
                    bias_pos = len(token_ids) + 1
                    positions += list(range(bias_pos, bias_pos + img_tokens_num))
                    token_num_start = len(token_ids)
                    token_ids += [img_start_id] + [img_token_id] * img_tokens_num + [img_end_id]
                    if xd_pos_num in [3, 4]:
                        token_num = len(token_ids)
                        position_num = len(position_ids_t)
                        patch_width = patch_width_list[img_count]
                        patch_height = (img_tokens_num - 2) // (patch_width + 1)
                        position_ids_t += [position_num, position_num + 1] + [img_count] * (img_tokens_num - 2) + [token_num - 2, token_num - 1]
                        position_ids_x += [position_num, position_num + 1] + list(range(patch_width+1)) * patch_height + [token_num - 2, token_num - 1]
                        position_ids_h = torch.arange(patch_height).reshape(-1,1).expand(patch_height, patch_width+1)
                        position_ids_h = position_ids_h.reshape(-1).numpy().tolist()
                        position_ids_y += [position_num, position_num + 1] + position_ids_h + [token_num - 2, token_num - 1]
                elif data_mode in [0, 1] and img_emb_list[img_count].shape[1] > 1:
                    token_num_start = len(token_ids)
                    token_ids += [img_start_id, 128167]
                    img_tokens_num = img_emb_list[img_count].shape[2]
                    for sub_index in range(img_emb_list[img_count].shape[1]):
                        bias_pos = len(token_ids)
                        positions += list(range(bias_pos, bias_pos + img_tokens_num))
                        token_ids += [img_token_id] * img_tokens_num + [128168]
                    token_ids += [128169, img_end_id]
                    if xd_pos_num in [3, 4]:
                        token_num = len(token_ids)
                        position_num = len(position_ids_t)
                        patch_width = img_emb_list[img_count].shape[2]
                        patch_height = img_emb_list[img_count].shape[1]
                        img_tokens_num = patch_height * (patch_width + 1) + 2
                        position_ids_t += [position_num, position_num + 1] + [img_count] * (img_tokens_num - 2) + [token_num - 2, token_num - 1]
                        position_ids_x += [position_num, position_num + 1] + list(range(patch_width+1)) * patch_height + [token_num - 2, token_num - 1]
                        position_ids_h = torch.arange(patch_height).reshape(-1,1).expand(patch_height, patch_width+1)
                        position_ids_h = position_ids_h.reshape(-1).numpy().tolist()
                        position_ids_y += [position_num, position_num + 1] + position_ids_h + [token_num - 2, token_num - 1]
                elif data_mode == 2 and img_emb_list[img_count].shape[1] > 1:
                    img_num = img_emb_list[img_count].shape[1]
                    if len(img_emb_list[0].shape) == 4:
                        img_tokens_num = img_emb_list[img_count].shape[2]
                        for img_index in range(img_num):
                            bias_pos = len(token_ids) + 1
                            positions += list(range(bias_pos, bias_pos + img_tokens_num))
                            token_num_start = len(token_ids)
                            token_ids += [img_start_id] + [img_token_id] * img_tokens_num + [img_end_id]
                    if len(img_emb_list[0].shape) == 5:
                        img_tokens_num = img_emb_list[img_count].shape[3]
                        for img_index in range(img_num):
                            token_num_start = len(token_ids)
                            token_ids += [img_start_id, 128167]
                            for sub_index in range(img_emb_list[img_count].shape[2]):
                                bias_pos = len(token_ids)
                                positions += list(range(bias_pos, bias_pos + img_tokens_num))
                                token_ids += [img_token_id] * img_tokens_num + [128168]
                            token_ids += [128169, img_end_id]
                elif data_mode == 3 and img_emb_list[img_count].shape[1] > 1:
                    img_num = img_emb_list[img_count].shape[1]
                    img_tokens_num = img_emb_list[img_count].shape[2]
                    for img_index in range(img_num):
                        token_ids += [img_start_id]
                        bias_pos = len(token_ids)
                        positions += list(range(bias_pos, bias_pos + img_tokens_num))
                        token_ids += [img_token_id] * img_tokens_num
                        token_ids += [img_end_id]
                        if xd_pos_num in [3, 4]:
                            patch_width = patch_width_list[img_count]
                            patch_height = (img_tokens_num - 2) // (patch_width + 1)
                            position_ids_x += [bias_pos - 1, bias_pos]
                            position_ids_y += [bias_pos - 1, bias_pos]
                            position_ids_t += [bias_pos - 1, bias_pos]

                            position_ids_x += list(range(patch_width + 1)) * patch_height
                            for i in range(patch_height):
                                position_ids_y += [i] * (patch_width + 1)
                            position_ids_t += [img_index] * (img_tokens_num - 2)

                            bias_pos = len(token_ids)
                            position_ids_x += [bias_pos - 2, bias_pos - 1]
                            position_ids_y += [bias_pos - 2, bias_pos - 1]
                            position_ids_t += [bias_pos - 2, bias_pos - 1]

                if len(img_emb_list[0].shape) == 4:
                    b, h, w, c = img_emb_list[img_count].shape
                if len(img_emb_list[0].shape) == 5:
                    b, mb, h, w, c = img_emb_list[img_count].shape
                    h = mb * h
                img_emb_list[img_count] = img_emb_list[img_count].reshape(b, h*w, 1, c)

                img_count += 1
        if isinstance(input_info, str):
            if data_mode in [1, 2, 3] and len(positions) > 0:
                token_ids += [video_end_id]
                episode_flag = True
                if xd_pos_num in [3, 4]:
                    token_num = len(token_ids)
                    position_ids_t += [token_num - 1]
                    position_ids_x += [token_num - 1]
                    position_ids_y += [token_num - 1]
            img_id_len = len(token_ids)
            input_info = replace_special_tokens(input_info, token_map)
            str_ids = tokenizer.tokenizer.encode(input_info)
            if isinstance(str_ids, list):
                token_ids += str_ids
            else:
                token_ids += str_ids.ids

    # if exact_tokenizer:
    #     if tokenizer.prompt_postfix:
    #         token_ids += tokenizer.tokenizer.encode(tokenizer.prompt_postfix)
    #     if tokenizer.prompt_add_sep:
    #         token_ids = token_ids + [sep_token_id]

    token_ids += [sep_token_id]

    if xd_pos_num in [3, 4]:
        token_num = len(token_ids)
        x_position_num = len(position_ids_t)
        position_ids = list(range(token_num))
        position_ids_t += position_ids[x_position_num:]
        position_ids_x += position_ids[x_position_num:]
        position_ids_y += position_ids[x_position_num:]

    if special_flag:
        token_ids += [290944, 705, 290943, 120321, 705]

    if len(token_ids) > max_input_seq_len:
        assert len(token_ids) <= max_input_seq_len, "token_ids_len: {}; input_info_len: {}; img_emb_list: {}; img_emb_shape: {}.".format(len(token_ids), len(input_infos), len(img_emb_list), img_emb_list[0].shape)
        token_ids = token_ids[:max_input_seq_len]
        if xd_pos_num in [3, 4]:
            position_ids   = position_ids[:max_input_seq_len]
            position_ids_t = position_ids_t[:max_input_seq_len]
            position_ids_x = position_ids_t[:max_input_seq_len]
            position_ids_y = position_ids_t[:max_input_seq_len]

    input_ids = np.array([token_ids]).astype(np.uint32)
    mem_seq_len = [len(token_ids)]
    mem_seq_len = np.array(mem_seq_len).reshape(-1,1).astype(np.uint32)
    img_pos = np.array([positions]).astype(np.int32)
    # if verbose:
    # print("input_ids:", token_ids, flush=True)
    # print("decode input_ids:", tokenizer.tokenizer.decode(token_ids), flush=True)
    if len(img_emb_list) > 0:
        img_emb = np.concatenate(img_emb_list, axis=1)

    if xd_pos_num == 3:
      rotary_xd_position_ids = np.array([[position_ids_x, position_ids_y, position_ids_t]], dtype=np.uint32)
      rotary_xd_rope_section = np.array([[0.25, 0.25, 0.5]], dtype=np.float32)
    if xd_pos_num == 4:
      rotary_xd_position_ids = np.array([[position_ids, position_ids_x, position_ids_y, position_ids_t]], dtype=np.uint32)
      rotary_xd_rope_section = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)

    if verbose:
        print(f'input_ids:\n{input_ids}')

    output_dict = {'output_text': ''}
    tmp_ids_list = []

    def stream_callback(start_time, output_dict, tmp_ids_list, result, error):
        if error:
            print(error, file=sys.stderr)
        else:
            if len(output_dict['output_text']) == 0:
                if verbose:
                    print(f"First token cost {time.perf_counter() - start_time:.2f}s", file=sys.stderr, flush=True)

            output_ids = result.as_numpy("output_ids")
            output_len = result.as_numpy("sequence_length")
            log_probs = result.as_numpy('cum_log_probs')
            output_log_probs = result.as_numpy('output_log_probs')

            max_index = log_probs[0].argmax() if log_probs is not None else 0
            start_index = 0 # exclude_input_in_output set "true"
            end_index = output_len[0, max_index] if output_len is not None else None
            left_ids = output_ids[0, max_index, start_index:end_index].tolist()

            if eos_token_id in left_ids:
                left_ids = left_ids[:left_ids.index(eos_token_id)]
            if eod_token_id in left_ids:
                left_ids = left_ids[:left_ids.index(eod_token_id)]
            if pad_token_id in left_ids:
                left_ids = left_ids[:left_ids.index(pad_token_id)]
            # add pre ids to decode
            if len(tmp_ids_list) > 0:
                for ids in reversed(tmp_ids_list):
                    left_ids = ids + left_ids
                tmp_ids_list.clear()

            # decode ids to text
            if exact_tokenizer:
                output_text = tokenizer.decode_output(left_ids)
            else:
                output_text = tokenizer.tokenizer.decode(left_ids)
            # check output diamond
            # if has_output_diamond_question_mark(output_text):
            #     tmp_ids_list.append(left_ids)
            #     return

            output_dict['output_text'] += output_text
            output_dict['log_probs'] = log_probs[0, max_index] if log_probs is not None else 0.0

            if return_last_context_embeddings:
                last_context_embeddings = result.as_numpy("last_context_embeddings")
                output_dict['last_context_embeddings'] = last_context_embeddings

            if verbose:
                print("[After {:.2f}s] (probability {:.2e}) \n{}\n".format(
                    time.perf_counter() - start_time,
                    log_probs[0, max_index] if log_probs is not None else 0.0,
                    output_dict['output_text']))

                if verbose:
                    print('Output<:', file=sys.stderr, flush=True)
    with grpcclient.InferenceServerClient(url, verbose=False) as cl:
        input_ids = input_ids.astype(np.uint32)
        mem_seq_len = mem_seq_len.astype(np.uint32)

        inputs = [
            prepare_tensor(grpcclient, "input_ids", input_ids),
            prepare_tensor(grpcclient, "input_lengths", mem_seq_len)
        ]
        if len(img_emb_list) > 0:
            inputs += [
                prepare_tensor(grpcclient, "image_embeddings", img_emb),
                prepare_tensor(grpcclient, "image_positions", img_pos),
            ]

        inputs += prepare_trt_inputs(
                    cl, model_name, input_ids.shape[0], eos_token_id, output_seq_len,
                    beam_size, top_k, top_p, random_seed, diversity_rate,
                    temperature, len_penalty, repetition_penalty, min_length,
                    bad_words_list=bad_words_list, stop_words_list=stop_words_list,
                    return_log_probs=return_log_probs, return_top_log_probs=return_top_log_probs,
                    return_logits=return_logits, return_hidden_states=return_hidden_states,
                    return_last_context_embeddings=return_last_context_embeddings,
                    vocab_size=vocab_size, logit_bias=logit_bias,
                    repetition_ngrams=repetition_ngrams, repetition_patiences=repetition_patiences,
                    opt_request_output_len=opt_request_output_len, opt_request_keep_iter=opt_request_keep_iter,
                    is_streaming=stream_mode, valid_range_start_id=valid_range_start_id,
                    valid_range_end_id=valid_range_end_id, special_id=special_id,
                    max_special_id_position=max_special_id_position)

        if xd_pos_num in [3, 4]:
            inputs.append(prepare_tensor(grpcclient, "rotary_xd_position_ids", rotary_xd_position_ids))
            inputs.append(prepare_tensor(grpcclient, "rotary_xd_rope_section", rotary_xd_rope_section))

        cl.start_stream(callback=partial(stream_callback, time.perf_counter(), output_dict, tmp_ids_list))
        cl.async_stream_infer(model_name, inputs, request_id=str(request_id))

    if verbose:
        print('\n', file=sys.stderr, flush=True)

    output_dict["output_text"] = replace_special_tokens(output_dict["output_text"], token_map_inv)
    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="gpt server ip")
    parser.add_argument("--port", type=int, default=8021, help="gRPC server port")
    parser.add_argument("--vit_ip", type=str, default="localhost", help="vit server ip")
    parser.add_argument("--vit_port", type=int, default=9021, help="VIT server port")
    parser.add_argument("--model_name", type=str, default="gpt", help="gpt model name")
    parser.add_argument("--tokenizer_type", type=str, default="hy_ptm_for_business_gqa", help="tokenizer type")
    parser.add_argument("--exact_tokenizer", action="store_true", default=False, help="use exact tokenizer in text handler")
    parser.add_argument("--request_id", type=str, default="0", help="request id")
    parser.add_argument('--inputs', type=str, nargs='+', default=['./imgs/2.webp',
        '详细描述一下图像中的内容'], required=False)
    parser.add_argument('--output_len', type=int, default=128, required=False)
    parser.add_argument('--top_k', type=int, default=40, required=False)
    parser.add_argument('--min_length', type=int, default=1, required=False)
    parser.add_argument("--return_last_context_embeddings", action="store_true", default=False)
    parser.add_argument("--return_context_cum_log_probs", action="store_true", default=False,
            help="return context cum log probs")
    parser.add_argument("--xd_pos_num", type=int, default=0, help="3: (x, y, t); 4: (p, x, y, t)")
    parser.add_argument("--data_mode", type=int, default=0, help="0: image, 1: video, 2: video(merged multi-images), 3: video(with audio)")
    parser.add_argument("--video_spatial_compress", action="store_true", default=False, help="video spatial compress in vit")
    parser.add_argument("--video_temporal_compress", action="store_true", default=False, help="video temporal compress in vit")
    parser.add_argument("--verbose", action="store_true", default=False, help="print more info")
    parser.add_argument("--debug", action="store_true", default=False, help="debug with ipython")
    parser.add_argument('--valid_range_start_id', type=int, default=0, required=False)
    parser.add_argument('--valid_range_end_id', type=int, default=0, required=False)
    parser.add_argument('--special_id', type=int, default=-1, required=False)
    parser.add_argument('--max_special_id_position', type=int, default=-1, required=False)
    args = parser.parse_args()

    resolution, patch_num, max_image_num, img_tokens_num, patch_version = get_vit_infos(f'{args.vit_ip}:{args.vit_port}', 'vit')

    start_time = time.time()
    input_infos = []
    img_emb_list = []
    patch_width_list = []
    input_infos = parse_list_inputs(args.inputs, args.data_mode==2)
    for i, input_info in enumerate(input_infos):
        if isinstance(input_info, str):
            if input_info.startswith('http'):
                img_data = np.frombuffer(input_info.encode(), np.uint8)
            else:
                img_data = None
        elif isinstance(input_info, dict) and 'image_path' in input_info:
            img_pathes = input_info['image_path']
            if isinstance(img_pathes, str):
                img_path = img_pathes
                with open(img_path, 'rb') as f:
                    img_data = np.frombuffer(base64.b64encode(f.read()), np.uint8)
            elif isinstance(img_pathes, list):
                img_data_base64_list = []
                for img_path in img_pathes:
                    with open(img_path, 'rb') as f:
                        img_data_base64_list.append(base64.b64encode(f.read()))
                img_data = np.frombuffer(b" ".join(img_data_base64_list), np.uint8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if args.data_mode in [1, 2, 3] and args.video_spatial_compress and not args.video_temporal_compress:
            data_mode = 1
        elif args.data_mode in [1, 2, 3] and not args.video_spatial_compress and args.video_temporal_compress:
            # only support multi-images or video-files
            data_mode = 2
        elif args.data_mode in [1, 2] and args.video_spatial_compress and args.video_temporal_compress:
            # only support multi-images or video-files
            data_mode = 3
        else:
            data_mode = 0
        img_emb, patch_num_width = grpc_infer_vit(f'{args.vit_ip}:{args.vit_port}', 'vit',
                img_data, data_mode=data_mode,
                request_id=args.request_id+ f'_{i}')
        if img_emb.size > 0:
            # need use torch
            if img_emb.dtype == np.int16:
                img_emb = torch.from_numpy(img_emb).view(torch.bfloat16).float().numpy()
        img_emb_list.append(img_emb)
        patch_width_list.append(patch_num_width)

    mid_time = time.time()
    print(input_infos)

    for img_emb in img_emb_list:
        if img_emb.size > 0 and len(img_emb.shape) == 4:
            print(f'img_tokens_num: {img_emb.shape[1]} * {img_emb.shape[2]}')
        if img_emb.size > 0 and len(img_emb.shape) == 5:
            print(f'img_tokens_num: {img_emb.shape[1]} * {img_emb.shape[2]} * {img_emb.shape[3]}')
    output_grpc = grpc_stream_infer(
            f'{args.ip}:{args.port}', args.model_name,
            input_infos, img_emb_list, patch_width_list,
            output_seq_len=args.output_len,
            tokenizer_type=args.tokenizer_type,
            exact_tokenizer=args.exact_tokenizer,
            return_last_context_embeddings=args.return_last_context_embeddings,
            return_log_probs=args.return_context_cum_log_probs,
            bad_words_list = [[128167, 128168, 128169], [1, 2, 3]],
            data_mode=args.data_mode,
            top_k=args.top_k,
            min_length=args.min_length,
            verbose=args.verbose,
            xd_pos_num=args.xd_pos_num,
            valid_range_start_id=args.valid_range_start_id,
            valid_range_end_id=args.valid_range_end_id,
            special_id=args.special_id,
            max_special_id_position=args.max_special_id_position)
    end_time = time.time()
    print(output_grpc)
    print(f'time: {end_time - start_time: .3f}s | {mid_time - start_time: .3f}s  | {end_time - mid_time: .3f}s')

    if args.debug:
        from IPython import embed; embed()
        import sys; sys.exit()
