#!venv/bin/python
# from rembg import remove, new_session
import os
import glob
import io
import argparse
from PIL import Image, ImageColor
import numpy as np
import cv2
import contextlib

# Prevent DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

def process_image(input_path, output_path, bg_color=None, model_name="isnet-general-use", alpha_matting=False, af=240, ab=10, ae=10, post_process_mask=False, remove_color=None, tolerance=30, remove_watermark=False, use_gpu=False):
    try:
        # If output path is a directory (in case user messed up args), handle that
        if os.path.isdir(output_path):
            filename = os.path.basename(input_path)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_path, f"{name}.png")

        print(f"Loading image from: {input_path}")
        
        if remove_color:
            print(f"Using Color Removal Mode (Non-AI). Removing color: {remove_color} with tolerance {tolerance}")
            img_subject = Image.open(input_path).convert("RGBA")
            
            # Get target RGB
            try:
                target_rgb = ImageColor.getrgb(remove_color)
            except ValueError:
                print(f"Error: Invalid remove color format '{remove_color}'.")
                return

            # Convert to numpy array (RGB)
            img_np = np.array(img_subject)
            
            # Helper to check if a color is close to target
            # But for contiguous, we better use OpenCV floodFill
            
            # Convert to BGR for OpenCV
            bgr_image = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            
            # Create a mask for floodFill (h+2, w+2)
            h, w = bgr_image.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            # Parse target color to BGR
            # target_rgb from ImageColor is (R, G, B)
            target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
            
            # Define tolerance
            # cv2.floodFill takes lowerDiff and upDiff as tuples
            tol_val = (tolerance, tolerance, tolerance)
            
            # We want to fill the background with a specific value in the mask
            # flags: 
            # 4 or 8 connectivity
            # FLOODFILL_MASK_ONLY: we don't change the image, just the mask
            # (255 << 8): fill mask with 255
            # FLOODFILL_FIXED_RANGE: compare to seed color, not neighbor. Prevents "walking" into gradients.
            flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
            
            # Seed points: checks current background color at corners matches target?
            # Or just blindly flood fill from corners if they match the "remove_color" roughly.
            # Actually, standard approach:
            # 1. Check if corners are close to target color. 
            # 2. If yes, flood fill from there.
            
            corns = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
            
            flooded_any = False
            for x, y in corns:
                # Check pixel color
                pixel = bgr_image[y, x]
                # Calculate diff
                dist = np.sqrt(np.sum((pixel - target_bgr)**2))
                
                # If pixel is close enough to target color, start flood fill
                # Use a slightly looser check for the seed point itself to be safe, 
                # or just let floodFill handle it.
                if dist < tolerance * 2: # heuristic check
                     cv2.floodFill(bgr_image, mask, (x, y), (0,0,0), tol_val, tol_val, flags)
                     flooded_any = True

            if not flooded_any:
                 print("Warning: corners do not match the target removal color. Trying to find any matching pixel on border...")
                 # Fallback: scan top row
                 for x in range(0, w, 10):
                     pixel = bgr_image[0, x]
                     dist = np.sqrt(np.sum((pixel - target_bgr)**2))
                     if dist < tolerance:
                          cv2.floodFill(bgr_image, mask, (x, 0), (0,0,0), tol_val, tol_val, flags)
                          flooded_any = True
                          break
            
            # Mask is now 255 where background is found.
            # Mask size is (h+2, w+2), we need inner (h, w)
            final_mask = mask[1:-1, 1:-1]
            
            # Apply transparent alpha to flooded areas
            # img_np is RGBA
            # Where mask == 255, set Alpha (index 3) to 0
            img_np[final_mask == 255, 3] = 0
            
            img_subject = Image.fromarray(img_np)
            
        else:
            with open(input_path, 'rb') as i:
                input_data = i.read()
                
            print(f"Removing background using model '{model_name}'...")
            
            # Lazy import to prevent hang on manual mode
            from rembg import remove, new_session
            
            if model_name in ["sam", "birefnet-general", "isnet-anime"]:
                print("Note: This model is very large (hundreds of MBs to 1GB+).")
                print("If this is the first time running, it may look like it's hanging while downloading. Please wait...")
            
            if alpha_matting:
                print(f"Alpha matting enabled (fg_thresh={af}, bg_thresh={ab}, erode={ae})...")

            # Force CPU provider to prevent hangs on newer numpy/macos combos
            # Also set OMP_NUM_THREADS to avoid thread contention
            os.environ["OMP_NUM_THREADS"] = "1"
            
            if use_gpu:
                print(f"Creating session for model '{model_name}' with GPU/Auto providers...")
                # If GPU is requested, we pass None to let rembg/ORT use available providers (likely including CoreML/CUDA)
                session = new_session(model_name, providers=None)
            else:
                print(f"Creating session for model '{model_name}' with CPUExecutionProvider...")
                session = new_session(model_name, providers=['CPUExecutionProvider'])

            print("Session created successfully.")
            
            print("Starting background removal process... (this might take a moment)")
            # Suppress pymatting performance warnings that clutter stdout
            with contextlib.redirect_stdout(io.StringIO()):
                subject = remove(
                    input_data, 
                    session=session,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=af,
                    alpha_matting_background_threshold=ab,
                    alpha_matting_erode_size=ae,
                    post_process_mask=post_process_mask
                )
            # print("Background removal complete.")
                
            img_subject = Image.open(io.BytesIO(subject)).convert("RGBA")
        
        final_image = img_subject
        
        if bg_color:
            print(f"Applying background color: {bg_color}")
            try:
                # Check if the color is valid
                color = ImageColor.getrgb(bg_color)
                # Create a solid color image
                background = Image.new("RGBA", img_subject.size, color)
                # Paste the subject on top of the background (using the subject itself as mask)
                background.paste(img_subject, (0, 0), img_subject)
                
                # Check output format
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    final_image = background.convert("RGB")
                    print("Output format is JPEG, saving as RGB (no transparency).")
                else:
                    final_image = background
            except ValueError:
                print(f"Error: Invalid color format '{bg_color}'. Use standard names (e.g. 'black', 'white') or hex codes (e.g. '#FF0000').")
                return
        else:
            print("No background color specified. Saving with transparent background.")
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                 print("Warning: Saving as JPEG without a specified background color will lose transparency (usually appearing black).")
                 final_image = img_subject.convert("RGB")

        if remove_watermark:
            print("Removing watermark from bottom-right corner...")
            # Assuming standard Google Gemini/AI watermark size relative to image
            # Or just a fixed safe area. 
            # A simple approach is to mask out a small rectangle in the bottom right
            # RGB 255,255,255 usually, or similar to background.
            # Since we are removing background, ensuring transparency there is key.
            
            # Convert back to numpy to edit
            data = np.array(final_image)
            rows, cols = data.shape[:2]
            
            # Define region size (e.g. 170x60 pixels seems common for "Generated with Google")
            # But let's be proportional to be safe, or fixed if we know it.
            # Fixed 200x80 pixels from bottom right
            wm_w = 240
            wm_h = 240
            
            # Ensure we don't go out of bounds
            if rows > wm_h and cols > wm_w:
                # Set alpha to 0 in that region
                data[rows-wm_h:rows, cols-wm_w:cols, 3] = 0
                final_image = Image.fromarray(data)
                
        print(f"Saving output to: {output_path}")
        final_image.save(output_path)
        print("Done!")

    except Exception as e:
        print(f"An error occurred processing {input_path}: {e}")

def process_batch(input_dir, output_dir, bg_color=None, model_name="isnet-general-use", alpha_matting=False, af=240, ab=10, ae=10, post_process_mask=False, remove_color=None, tolerance=30, remove_watermark=False, use_gpu=False):
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        # Also check for uppercase extensions
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    files = sorted(list(set(files)))
    total_files = len(files)
    
    if total_files == 0:
        print(f"No image files found in '{input_dir}'.")
        return

    print(f"Found {total_files} images to process in batch mode.")

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        name_part, ext_part = os.path.splitext(filename)
        
        # Determine output filename
        if not bg_color:
            new_filename = name_part + ".png"
        else:
             # Keep original extension if color provided, unless user wants specific format handling logic
             # For simplicity, if input is jpg and color is white, saving as jpg is fine.
             # But if input is png, saving as png is fine.
             new_filename = filename
        
        output_path = os.path.join(output_dir, new_filename)
        
        print(f"\n[{i+1}/{total_files}] Processing: {filename}")
        process_image(
            file_path, 
            output_path, 
            bg_color, 
            model_name=model_name,
            alpha_matting=alpha_matting,
            af=af,
            ab=ab,
            ae=ae,
            post_process_mask=post_process_mask,
            remove_color=remove_color,
            tolerance=tolerance,
            remove_watermark=remove_watermark,
            use_gpu=use_gpu
        )
    
    print(f"\nBatch processing complete! All files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Remove background from an image or a folder of images.")
    parser.add_argument("input", help="Path to input image file OR directory of images")
    parser.add_argument("-o", "--output", required=True, help="Path to output image file OR output directory (if input is a directory)")
    parser.add_argument("-c", "--color", help="Background color (e.g., 'black', 'white', '#FF0000'). If omitted, background will be transparent.")
    
    # Advanced options
    parser.add_argument("--model", default="isnet-general-use", help="Model to use (default: isnet-general-use). Options: u2net, isnet-general-use, isnet-anime, u2net_human_seg, etc.")
    parser.add_argument("-a", "--alpha-matting", action="store_true", help="Enable alpha matting for smoother edges.")
    parser.add_argument("--af", type=int, default=240, help="Alpha matting foreground threshold (default: 240)")
    parser.add_argument("--ab", type=int, default=10, help="Alpha matting background threshold (default: 10)")
    parser.add_argument("--ae", type=int, default=10, help="Alpha matting erode size (default: 10)")
    parser.add_argument("-p", "--post-process-mask", action="store_true", help="Enable post-processing mask (good for cleaner edges, but slower).")
    
    # Chroma Key Options (Non-AI)
    parser.add_argument("-rc", "--remove-color", help="Remove a specific color (e.g. 'black', '#000000') instead of using AI. Good for solid backgrounds.")
    parser.add_argument("-t", "--tolerance", type=int, default=30, help="Color tolerance for removal (0-255). Default: 30.")
    parser.add_argument("-w", "--remove-watermark", action="store_true", help="Remove watermark from bottom-right corner (common in AI generated images).")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (experimental, defaults to CPU only).")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        # Batch Mode
        process_batch(
            args.input, 
            args.output, 
            args.color, 
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            af=args.af,
            ab=args.ab,
            ae=args.ae,
            post_process_mask=args.post_process_mask,
            remove_color=args.remove_color,
            tolerance=args.tolerance,
            remove_watermark=args.remove_watermark,
            use_gpu=args.gpu
        )
    else:
        # Single File Mode
        process_image(
            args.input, 
            args.output, 
            args.color, 
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            af=args.af,
            ab=args.ab,
            ae=args.ae,
            post_process_mask=args.post_process_mask,
            remove_color=args.remove_color,
            tolerance=args.tolerance,
            remove_watermark=args.remove_watermark,
            use_gpu=args.gpu
        )

if __name__ == "__main__":
    main()
