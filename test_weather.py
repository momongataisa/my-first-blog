import os
import random
import argparse
import tarfile
import time
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
    tar_path = args.source_file
    
    root_dir = Path("/root/working/ito_tatsuto/CAM_FRONT")
    print(root_dir)
    
    # 入力ファイルの解凍
    if tarfile.is_tarfile(tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=root_dir)
            print(f"ファイルを解凍しました：{root_dir}")
    else:
        print(f"{tar_path}は有効な.tar.gzファイルではない。")
    
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )


    # processorのインストール
    min_pixels = 256*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
   
    file_dir_nuscenes = root_dir / "CAM_FRONT"
    file_paths_nuscenes = os.listdir(file_dir_nuscenes)
    file_paths_nuscenes = [file_dir_nuscenes / file_path for file_path in file_paths_nuscenes]
    print(len(file_paths_nuscenes))
    
    # すべてのデータを統合
    all_paths = file_paths_nuscenes
    # file_paths_mini = random.sample(all_paths, 5)
    file_paths_mini = all_paths
    
    # 天気の判定
    text = """What is the weather in [location] on [date/time]? Please respond with a number corresponding to the following:
    - Sunny: 0
    - Cloudy: 1
    - Rainy: 2
    - Other weather (e.g., snow, storm, fog): 3"""
    
    # file_paths, textをもとにQwen2に入力するためのメッセージの作成
    messages = make_messages(file_paths_mini, text)
    
    # 作成したメッセージをバッチサイズに変更
    batch_messages, batch_file_paths = change_messages_to_batch(messages, file_paths_mini, 2)
    
    output_list = []
    
    start = time.time()
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
            
    end = time.time()
    
    print(f"処理時間：{end-start}")
    
    
    args.output.mkdir(exist_ok=True)  # フォルダが存在しない場合のみ作成
    new_sunny = args.output  / "sunny"
    new_cloudy = args.output / "cloudy"
    new_rainy = args.output / "rainy"
    new_another = args.output / "another"
    # サブフォルダを作成（os.makedirs を使う場合）
    new_sunny.mkdir(exist_ok=True, parents=True)
    new_cloudy.mkdir(exist_ok=True, parents=True)
    new_rainy.mkdir(exist_ok=True, parents=True)
    new_another.mkdir(exist_ok=True, parents=True)
    
    # 画像を適切なフォルダに保存
    for i, output in enumerate(output_list):
        img_path, label = output  # output[0] は画像、output[1] はラベル
        img = Image.open(img_path)
        label = label.lower()
        
        # 画像のファイル名を決める
        img_filename = str(img_path).split('/')[-1]  # 連番ファイル名

        
        # 保存先フォルダを決定
        if "0" in label or "sunny" in label:
            save_path = new_sunny / img_filename
        elif "1" in label or "cloudy" in label:
            save_path = new_cloudy / img_filename
        elif "2" in label or "rainy" in label:
            save_path = new_rainy / img_filename
        else:
            save_path = new_another /img_filename

        # 画像を保存
        img.save(save_path)
        print(f"画像を {save_path} に保存")
        
    with open(args.output/"output.txt", "w") as f:
        for output in output_list:
            f.write(f"path: {output[0]}, answer: {output[1]}\n")
    print(f"PathとLabelを保存完了")
    print(f"推論時間：{end-start}")
        
if __name__ == '__main__':
    main()