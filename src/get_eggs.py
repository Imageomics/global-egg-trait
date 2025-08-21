from detect_segment_crop import *

def process_csv_with_pipeline(
    input_csv,
    config_yaml,
    crop_prefix="crop",
    verbose=True,
    embed=False
):
    # Load CSV and YAML config
    df = pd.read_csv(input_csv)
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "./output")
    output_csv = config.get("output_csv", "./output/results.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output records
    output_records = []
    
    device = "cuda"
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

    for idx, row in df.iterrows():
        image_path = row[config.get("filepath_col", "image_path")]
        uuid = row["uuid"]
        # Extract config values
        text_prompt = config.get("text_prompt", "")
        sam2_checkpoint = config["sam2_checkpoint"]
        model_cfg = config["model_cfg"]
        coco_json_path = config.get("coco_json_path", f"{output_dir}/nms_boxes_{idx}.json")
        device = config.get("device", None)
        visualize = config.get("visualize", False)

        if verbose:
            print(f"Processing row {idx+1}/{len(df)}: {image_path}, with prompt '{text_prompt}'")

        # Run pipeline
        crops, all_label_masks_squeezed, all_labels_scores, nms_boxes, nms_labels, nms_scores = detect_segment_crop_pipeline(
            image_path,
            text_prompt,
            processor,
            model,
            sam2_checkpoint,
            model_cfg,
            coco_json_path=coco_json_path,
            device=device,
            visualize=visualize
        )
        
        if len(crops) == 0:
            if verbose:
                print(f"No crops found for {image_path}")
            # Still add a record to track processed images with no detections
            output_row = row.to_dict()
            output_row["crop_path"] = None
            output_records.append(output_row)
            continue

        # Embed crops
        if embed:
            embeddings = bioclip_embed(crops)

        # Save crops and prepare output rows
        for crop_idx, crop in enumerate(crops):
            #crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
            crop_filename = f"{crop_prefix}_{uuid}_{crop_idx}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path)
            # Prepare output row: duplicate metadata, add crop path and embedding
            output_row = row.to_dict()
            output_row["crop_path"] = crop_path
            output_records.append(output_row)
        # for crop_idx, (crop, emb) in enumerate(zip(crops, embeddings)):
        #     crop_filename = f"{crop_prefix}_{idx}_{crop_idx}.png"
        #     crop_path = os.path.join(output_dir, crop_filename)
        #     crop.save(crop_path)
        #     # Prepare output row: duplicate metadata, add crop path and embedding
        #     output_row = row.to_dict()
        #     output_row["crop_path"] = crop_path
        #     if embed:
        #         output_row["embedding"] = emb.tolist()  # Save as list for CSV compatibility
        #     output_records.append(output_row)

    # Write output CSV
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_csv> <config_yaml>")
        sys.exit(1)

    input_csv = sys.argv[1]
    config_yaml = sys.argv[2]

    process_csv_with_pipeline(
        input_csv,
        config_yaml
    )
