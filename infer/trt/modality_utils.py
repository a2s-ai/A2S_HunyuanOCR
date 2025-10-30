import os
import io
import time
import numpy as np
import json
import base64
import requests
import chardet
import threading
from PIL import Image
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.grpc.service_pb2 import ModelInferResponse

from LLMTextHandler.tritonft.utils import prepare_tensor
from LLMTextHandler.handler_manager import get_handler


def base64_encode_image(image_file):
    with open(image_file, 'rb') as f:
        image_bytes = f.read()
    image_encode = base64.b64encode(image_bytes)
    image_encode = image_encode.decode('ascii')

    return image_encode


def parse_list_inputs(inputs, merge_media=False):
    input_infos = []
    img_pathes = []
    # fake inputs
    inputs.append('')
    for input_text in inputs:
        input_text = input_text.replace("\\n", "\n")
        if os.path.isfile(input_text):
            if merge_media:
                img_pathes.append(input_text)
            else:
                input_infos.append({'image_path': input_text})
        elif os.path.isdir(input_text):
            file_list = os.listdir(input_text)
            img_list = [file_name for file_name in file_list if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')]
            img_list.sort()
            if merge_media:
                for img_file in img_list:
                    img_pathes.append(os.path.join(input_text, img_file))
            else:
                for img_file in img_list:
                    input_infos.append({'image_path': os.path.join(input_text, img_file)})
        else:
            if len(img_pathes) > 0:
                input_infos.append({'image_path': img_pathes})
                img_pathes = []
            if len(input_text) > 0:
                input_infos.append(input_text)
    inputs.remove('')

    return input_infos


def flask_infer_vit(ip, img_path):
    params = {'image': base64_encode_image(img_path)}
    headers = {'content-type': 'application/json','Connection': 'close'}
    response = requests.post(f'http://{ip}:20000', headers=headers, json=params)
    json_out = json.loads(response.text.strip())
    img_emb = np.array([json_out['image_embedding']]).astype(np.float32)

    return img_emb


def get_vit_infos(url, model_name):
    with grpcclient.InferenceServerClient(url, verbose=False) as client:
        config = client.get_model_config(model_name, as_json=True)['config']
        resolution = int(config['parameters']['resolution']['string_value'])
        patch_num = 1
        if 'patch' in config['parameters']:
            patch_num = int(config['parameters']['patch']['string_value'])
        img_tokens_num = int(config['output'][0]['dims'][1])
        patch_version = 'NA'
        if 'patch_version' in config['parameters']:
            patch_version = config['parameters']['patch_version']['string_value']
        if patch_num == 1:
            max_image_num = 1
        elif patch_version=='V1':
            max_image_num = patch_num ** 2 + 1
        elif patch_version=='V2':
            max_image_num = patch_num ** 2 + 4
        elif patch_version=='V3' or patch_version == 'V4':
            max_image_num = patch_num + 4
        else:
            raise NotImplementedError

        img_tokens_num = 4 * img_tokens_num if (patch_version in ["V2", "V3", "V4"]) else img_tokens_num

    return resolution, patch_num, max_image_num, img_tokens_num, patch_version


def grpc_infer_vit(url, model_name, img_data=None, img_index=None,
        data_mode=0, width=None, height=None, request_id=0):
    if img_data is not None:
        img_data = np.expand_dims(img_data, axis=0)

    input_resolution = None
    if width is not None and height is not None:
        input_resolution = np.array([[width, height]], dtype=np.int32)

    patch_num_width = None
    with grpcclient.InferenceServerClient(url, verbose=False) as client:
        if img_data is not None:
            inputs = [prepare_tensor(grpcclient, "INPUT_0", img_data),]
            if img_index is not None:
                index_data = np.array([[img_index]], dtype=np.int32)
                inputs.append(prepare_tensor(grpcclient, "img_index", index_data))
            if input_resolution is not None:
                inputs.append(prepare_tensor(grpcclient, "input_resolution", input_resolution))
            if data_mode:
                data_mode = np.array([[data_mode]], dtype=np.int32)
                inputs.append(prepare_tensor(grpcclient, "data_mode", data_mode))
            result = client.infer(model_name, inputs, request_id=str(request_id))
            img_emb = result.as_numpy('OUTPUT_0')
            patch_num_width = result.as_numpy("patch_num_width")
        else:
            model_config = client.get_model_config(model_name, as_json=True)['config']
            return_min_size = 0
            if 'return_min_size' in model_config['parameters']:
                return_min_size = int(model_config['parameters']['return_min_size']['string_value'])
            output_dims = model_config['output'][0]['dims']
            last_dim = return_min_size if return_min_size > 0 else output_dims[-1]
            img_emb = np.zeros([1, 0, max(int(output_dims[-2]), 1), int(last_dim)], np.float32)

    if patch_num_width is None:
        return img_emb, None
    else:
        return img_emb, patch_num_width[0]


def multi_patch(img_file, patch_num=1, resolution=448, patch_version="V1"):
    img = Image.open(img_file).convert("RGB")
    width, height = img.width, img.height

    if patch_version == "V1":
        new_width, new_height = width, height
        if width >= height and width > resolution * patch_num:
            new_width = resolution * patch_num
            new_height = height * resolution * patch_num / width
        elif height > width and height > resolution * patch_num:
            new_height = resolution * patch_num
            new_width = width * resolution * patch_num / height

        patch_num_width = int((new_width - 0.1) / resolution) + 1
        patch_num_height = int((new_height - 0.1) / resolution) + 1
    elif patch_version == "V2":
        img_max_length = max(width, height)
        img_min_length = min(width, height)

        max_patch = min(patch_num, int((img_max_length - 0.1) / resolution) + 1)
        min_patch = int(img_min_length * max_patch / img_max_length- 0.0001 ) + 1

        (patch_num_width, patch_num_height) = (max_patch, min_patch) if width >= height else (min_patch, max_patch)
    elif patch_version == "V3" or patch_version == "V4":
        img_max_length = max(width, height)
        img_min_length = min(width, height)
        patch_num = 16
        if patch_version == "V4":
            patch_num = 64
        for length in range(patch_num,3,-1):
            # 先计算长边切patch长度，再将图片等比例缩放，然后再切短边
            max_patch = min(length, int((img_max_length - 0.1) / resolution) + 1)
            min_patch = int(img_min_length * max_patch / img_max_length - 0.0000001) + 1
            if max_patch * min_patch <= patch_num:
                (patch_num_width, patch_num_height) = (max_patch, min_patch) if width >= height else (min_patch, max_patch)
                break
    else:
        raise NotImplementedError

    img = img.resize((resolution * patch_num_width,
                      resolution * patch_num_height))

    img_list = []
    img_index = []

    if patch_version == "V2" or patch_version == "V1":
        for i in range(patch_num_height):
            for j in range(patch_num_width):
                left, top = j * resolution, i * resolution
                right, bottom = (j + 1) * resolution, (i + 1) * resolution
                sub_img = img.crop((left, top, right, bottom))
                img_bytes = io.BytesIO()
                sub_img.save(img_bytes, format='JPEG')
                img_data = np.frombuffer(base64.b64encode(img_bytes.getvalue()), np.uint8)
                img_list.append(img_data)
                img_index.append(j*patch_num+i+1)
        patch_num_width_fix = patch_num_height
    elif  patch_version == "V3" or patch_version == "V4":
        for i in range(patch_num_height):
            for j in range(patch_num_width):
                left, top = j * resolution, i * resolution
                right, bottom = (j + 1) * resolution, (i + 1) * resolution
                sub_img = img.crop((left, top, right, bottom))
                img_bytes = io.BytesIO()
                sub_img.save(img_bytes, format='JPEG')
                img_data = np.frombuffer(base64.b64encode(img_bytes.getvalue()), np.uint8)
                img_list.append(img_data)
                if patch_version == "V3":
                    img_index.append(i*patch_num+j+1)
                elif patch_version == "V4":
                    img_index.append(1)
        patch_num_width_fix = patch_num_width
    else:
        assert False

    if len(img_list) > 1:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = np.frombuffer(base64.b64encode(img_bytes.getvalue()), np.uint8)
        img_list.append(img_data)
        img_index.append(0)
    else:
        img_index=[0]

    return img_list, patch_num_width_fix, img_index


class ModelConfigurator:
    def __init__(self, client, model_name):
        if isinstance(client, httpclient.InferenceServerClient):
            self.model_config = client.get_model_config(model_name)
        elif isinstance(client, grpcclient.InferenceServerClient):
            self.model_config = client.get_model_config(model_name, as_json=True)['config']
        else:
            assert False, "Unkown triton client type, only support httpclient/grpcclient now"

        self.inputs = set()
        for input_dict in self.model_config['input']:
            self.inputs.add(input_dict['name'])

    def is_decoupled(self) -> bool:
        policy_dict = self.model_config['model_transaction_policy']
        if 'decoupled' not in policy_dict:
            return False
        return policy_dict['decoupled']

    def has_input(self, input_name) -> bool:
        return input_name in self.inputs

    def scheduler_policy(self) -> str:
        batch_scheduler_policy = self.model_config['parameters']['batch_scheduler_policy']
        if 'string_value' in batch_scheduler_policy:
            return self.model_config['parameters']['batch_scheduler_policy']['string_value']
        else:
            return ""


def has_output_diamond_question_mark(s):
    """ 判断s是否为乱码 """
    s_bytes = s.encode('utf-8')
    encoding = chardet.detect(s_bytes)['encoding']
    if encoding is None:
        return False
    try:
        s_decoded = s_bytes.decode(encoding)
        if '�' in s:
            return True
        else:
            return False
    except UnicodeDecodeError:
        return True


def prepare_trt_inputs(
        client, model_name, batch_size, eos_token_id, output_seq_len=512,
        beam_size=1, top_k=40, top_p=1.0, random_seed=1234, diversity_rate=0.0,
        temperature=1.0, len_penalty=0.0, repetition_penalty=1.0, min_length=0,
        bad_words_list=list(), stop_words_list=list(), return_log_probs=False,
        return_top_log_probs=0, return_logits=False, return_hidden_states=False,
        return_last_context_embeddings=False,
        vocab_size=0, logit_bias=dict(),
        repetition_ngrams=None, repetition_patiences=None,
        opt_request_output_len=-1, opt_request_keep_iter=200, is_streaming=None,
        valid_range_start_id=0, valid_range_end_id=0,
        special_id=-1, max_special_id_position=-1):
    model_config = ModelConfigurator(client, model_name)

    if isinstance(client, httpclient.InferenceServerClient):
        client_type = httpclient
        is_streaming = False
    elif isinstance(client, grpcclient.InferenceServerClient):
        client_type = grpcclient
        if is_streaming is None:
            is_streaming =  model_config.is_decoupled()
    else:
        assert False, "Unkown triton client type, only support httpclient/grpcclient now"

    request_seq_len = (output_seq_len * np.ones([batch_size, 1])).astype(np.uint32)
    end_ids = eos_token_id * np.ones([batch_size, 1]).astype(np.uint32)
    beam_width = (beam_size * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_temperature = temperature * np.ones([batch_size, 1]).astype(np.float32)
    runtime_top_k = (top_k * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_top_p = top_p * np.ones([batch_size, 1]).astype(np.float32)
    runtime_len_penalty = len_penalty * np.ones([batch_size, 1]).astype(np.float32)
    runtime_repetition_penalty = repetition_penalty * np.ones([batch_size, 1]).astype(np.float32)
    request_min_length = (min_length * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_random_seed = random_seed * np.ones([batch_size, 1]).astype(np.uint64)
    runtime_streaming = is_streaming * np.ones([batch_size, 1]).astype(bool)

    inputs = [
        prepare_tensor(client_type, "request_output_len", request_seq_len),
        prepare_tensor(client_type, "end_id", end_ids),
        prepare_tensor(client_type, "beam_width", beam_width),
        prepare_tensor(client_type, "temperature", runtime_temperature),
        prepare_tensor(client_type, "runtime_top_k", runtime_top_k),
        prepare_tensor(client_type, "runtime_top_p", runtime_top_p),
        prepare_tensor(client_type, "len_penalty", runtime_len_penalty),
        prepare_tensor(client_type, "repetition_penalty", runtime_repetition_penalty),
        prepare_tensor(client_type, "min_length", request_min_length),
        prepare_tensor(client_type, "random_seed", runtime_random_seed),
        prepare_tensor(client_type, "streaming", runtime_streaming),
    ]

    def append_if_not_none(var, name, dtype):
        if var is not None:
            np_array = var * np.ones([batch_size, 1]).astype(dtype)
            inputs.append(prepare_tensor(client_type, name, np_array))

    if model_config.has_input('bad_words_list'):
        if len(bad_words_list) > 0:
            runtime_bad_words_ids = np.array([bad_words_list] * batch_size, dtype=np.int32)
            inputs.append(prepare_tensor(client_type, "bad_words_list", runtime_bad_words_ids))
        if len(stop_words_list) > 0:
            runtime_stop_words_ids = np.array([stop_words_list] * batch_size, dtype=np.int32)
            inputs.append(prepare_tensor(client_type, "stop_words_list", runtime_stop_words_ids))

        if return_log_probs:
            append_if_not_none(return_log_probs, "return_log_probs", bool)

    if model_config.has_input('return_logits'):
        if return_log_probs and return_top_log_probs > 0:
            append_if_not_none(return_top_log_probs, "return_top_log_probs", np.uint32)

        if return_logits:
            append_if_not_none(return_logits, "return_logits", bool)

        if len(logit_bias) > 0 and vocab_size > 0:
            embedding_bias = np.zeros([batch_size, vocab_size], dtype=np.float32)
            for bs in range(batch_size):
                for k,v in logit_bias.items():
                    embedding_bias[bs][k] = v
            inputs.append(prepare_tensor(client_type, "embedding_bias", embedding_bias))

        append_if_not_none(repetition_ngrams, "repetition_ngrams", np.uint32)
        append_if_not_none(repetition_patiences, "repetition_patiences", np.uint32)

    if model_config.has_input('return_hidden_states'):
        if return_hidden_states:
            append_if_not_none(return_hidden_states, "return_hidden_states", bool)

    if model_config.has_input('return_last_context_embeddings'):
        if return_last_context_embeddings:
            append_if_not_none(return_last_context_embeddings, "return_last_context_embeddings", bool)

    batch_scheduler_policy = model_config.scheduler_policy()
    if batch_scheduler_policy == 'max_performance':
        append_if_not_none(opt_request_output_len, "opt_request_output_len", np.uint32)
        append_if_not_none(opt_request_keep_iter, "opt_request_keep_iter", np.uint32)

    if model_config.has_input('valid_range_start_id') and model_config.has_input('valid_range_end_id'):
        if not(valid_range_start_id == 0 and valid_range_end_id == 0):
            assert(valid_range_start_id < valid_range_end_id), \
                f"valid_range_end_id({valid_range_end_id}) must be larger than valid_range_start_id({valid_range_start_id})"
            append_if_not_none(valid_range_start_id, "valid_range_start_id", np.uint32)
            append_if_not_none(valid_range_end_id, "valid_range_end_id", np.uint32)

    if model_config.has_input('special_id') and model_config.has_input('max_special_id_position'):
        if special_id >=0 and max_special_id_position >= 0:
            append_if_not_none(special_id, "special_id", np.uint32)
            append_if_not_none(max_special_id_position, "max_special_id_position", np.uint32)

    return inputs
