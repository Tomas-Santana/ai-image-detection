import os
import webdataset as wds

def process_dataset():
    input_base = "DATAROOT"
    output_base = "DATAROOT_WDS"
    
    # Class to integer mapping used throughout train/test loaders.
    class_to_idx = {"nature": 1, "ai": 0}

    # Iterate over all datasets in DATAROOT
    if not os.path.exists(input_base):
        print(f"Directory {input_base} not found.")
        return

    datasets = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    
    for dataset in datasets:
        dataset_path = os.path.join(input_base, dataset)
        
        # Process train and val splits
        for split in ["train", "val"]:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
            
            # Create output directory for shards
            out_dir = os.path.join(output_base, dataset)
            os.makedirs(out_dir, exist_ok=True)
            
            # Will produce files like DATAROOT_WDS/imagenet_ai_0419_biggan/train-000000.tar
            pattern = os.path.join(out_dir, f"{split}-%06d.tar")
            
            # Using ShardWriter to automatically split into chunks of ~1000 samples or ~100MB
            with wds.ShardWriter(pattern, maxcount=1000) as sink: # type:ignore
                for cls_name, cls_label in class_to_idx.items():
                    cls_path = os.path.join(split_path, cls_name)
                    if not os.path.exists(cls_path):
                        continue
                    
                    # Gather all images in this class folder
                    for root, dirs, files in os.walk(cls_path):
                        for file in files:
                            # Filter standard image extensions
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                filepath = os.path.join(root, file)
                                
                                # Read binary content
                                with open(filepath, "rb") as stream:
                                    image_data = stream.read()
                                
                                # Use relative path or unique ID as key
                                ext = file.split('.')[-1]
                                key = f"{cls_name}/{file.replace('.' + ext, '')}"
                                
                                # Write to shard
                                sink.write({
                                    "__key__": key,
                                    ext.lower(): image_data,
                                    "cls": cls_label  # Save label directly
                                })
            print(f"Finished processing {dataset} - {split}")

if __name__ == "__main__":
    process_dataset()
