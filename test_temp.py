import os
import random
import argparse
import tarfile
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



# バッチ推論のためのmessagesの設定
def make_messages(file_paths, text):
    messages = []

    for file_path in file_paths:
        img = Image.open(file_path)
        # Create a dictionary for each message
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ],
            }
        ]
        # Append the dictionary to the list
        messages.append(message)
        
    
    return messages

# messagesをバッチサイズに変更
def change_messages_to_batch(messages, file_paths, batch_size=5):
    messages_list = []
    file_paths_list = []
    cnt = 0
    
    for i in range((len(messages)-1) // batch_size + 1):
        
        if i == len(messages) // batch_size:
            messages_list.append(messages[cnt:])
            file_paths_list.append(file_paths[cnt:])
        else:
            messages_list.append(messages[cnt: cnt+batch_size])
            file_paths_list.append(file_paths[cnt: cnt+batch_size])
        
        cnt += batch_size

    return messages_list, file_paths_list

def main():
    parser = argparse.ArgumentParser(description="copy inputs", allow_abbrev=False)
    parser.add_argument("--input", default=None)
    parser.add_argument("--input0", type=Path, required=True, dest="source_file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tempDir", type=Path, required=True)

    args = parser.parse_args()
    root_dir = args.source_file
    # root_dir = "/root/working/ito_tatsuto/CAM_FRONT_RIGHT"
    output_file = args.output / "output.txt"
    
    # # 入力ファイルの解凍
    # if tarfile.is_tarfile(root_path):
    #     with tarfile.open(root_path, 'r:gz') as tar:
    #         tar.extractall(path=root_dir)
    #         print(f"ファイルを解凍しました：{root_dir}")
    # else:
    #     print(f"{root_path}は有効な.tar.gzファイルではない。")
    
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )


    # processorのインストール
    min_pixels = 256*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
   
    file_dir_nuscenes = "./nuscenes-mini/CAM_FRONT_RIGHT/"
    file_paths_nuscenes = os.listdir(file_dir_nuscenes)
    file_paths_nuscenes = [file_dir_nuscenes + file_path for file_path in file_paths_nuscenes]
    print(len(file_paths_nuscenes))
    
    # すべてのデータを統合
    all_paths = file_paths_nuscenes
    file_paths_mini = random.sample(all_paths, 5)
    
    # 都市部か田舎部の判定
    text = "Answer Yes if the image taken is rural and No if it is urban."
    
    # file_paths, textをもとにQwen2に入力するためのメッセージの作成
    messages = make_messages(file_paths_mini, text)
    
    # 作成したメッセージをバッチサイズに変更
    batch_messages, batch_file_paths = change_messages_to_batch(messages, file_paths_mini, 1)
    
    output_list = []
    
    print("start search")
    for messages, file_paths in zip(batch_messages, batch_file_paths):
        # Preparation for batch inference
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    
    
        # 出力
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    
        for idx in range(len(file_paths)):
            output_list.append((file_paths[idx], output_texts[idx]))
    
    
    args.output.mkdir()
    with open(args.output/"output.txt", 'w') as f:
        for output in output_list:
            f.write(f"path: {output[0]},", f"answer: {output[1]}\n")
        
if __name__ == '__main__':
    main()