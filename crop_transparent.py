import os
import argparse
import glob
from PIL import Image, ImageChops

# Prevent DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

def trim_image(input_path, output_path, padding=20, tolerance=0, quality=95):
    try:
        img = Image.open(input_path).convert("RGBA")
        
        # Method 1: Check for transparency (alpha channel)
        # If the image has any transparency, we prioritize trimming based on that.
        # But a fully opaque image (like JPEG) will have a full alpha channel.
        # So we check if the alpha channel is uniform (all 255).
        
        extbuf = img.getextrema()
        # extbuf is a list of tuples (min, max) for each band. 
        # For RGBA: [(R_min, R_max), (G_min, G_max), (B_min, B_max), (A_min, A_max)]
        alpha_band_extrema = extbuf[3] 
        
        bbox = None
        
        # If there are transparent pixels (min alpha < 255), use alpha bbox
        if alpha_band_extrema[0] < 255:
            # Use a threshold for alpha to handle noise from background removers
            alpha = img.split()[3]
            # Threshold: any pixel with alpha > 10 is considered content
            # This is more robust than img.getbbox() which detects ANY non-zero pixel.
            mask = alpha.point(lambda p: 255 if p > 10 else 0)
            bbox = mask.getbbox()
            if bbox:
                print(f"Transparency detected. Content bounding box: {bbox}")
            else:
                print(f"Transparency detected, but no pixels found above alpha threshold.")
        else:
            print(f"No transparency detected in '{os.path.basename(input_path)}'. Trimming based on background color...")
            # Method 2: Solid background removal
            bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
            diff = ImageChops.difference(img, bg)
            # IMPORTANT: For opaque images, alpha difference is 0. getbbox() on RGBA uses alpha.
            # We must convert to RGB (or Gray) to see the color difference.
            diff = diff.convert("RGB")
            
            # If tolerance is > 0, we can threshold the difference
            # Note: We always apply a small default tolerance (5) for photos if not specified
            effective_tolerance = tolerance if tolerance > 0 else 5
            print(f"Applied color tolerance: {effective_tolerance}")
            diff = Image.eval(diff, lambda x: 0 if x <= effective_tolerance else 255)
            
            bbox = diff.getbbox()
            if bbox:
                print(f"Background removal box: {bbox}")

        if not bbox:
            print(f"Warning: Image '{input_path}' appears to be empty or uniform. Skipping.")
            return

        # Unpack bbox
        left, upper, right, lower = bbox
        
        # Original dimensions
        orig_width, orig_height = img.size

        # Apply padding with boundaries check (DO NOT exceed original size)
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(orig_width, right + padding)
        lower = min(orig_height, lower + padding)
        
        # Crop using standard PIL crop (stays within bounds)
        cropped_img = img.crop((left, upper, right, lower))
        
        print(f"Final dimensions (with {padding}px padding): {cropped_img.size}")
        
        # Save
        save_kwargs = {}
        # If saving as JPEG, convert back to RGB and set quality
        if output_path.lower().endswith(('.jpg', '.jpeg')):
             cropped_img = cropped_img.convert("RGB")
             save_kwargs["quality"] = quality
             save_kwargs["subsampling"] = 0 # Keep best color info
             print(f"Saving to: {output_path} (JPEG, Quality: {quality})")
        else:
             print(f"Saving to: {output_path}")

        cropped_img.save(output_path, **save_kwargs)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_batch(input_dir, output_dir, padding=20, tolerance=0, quality=95):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    extensions = ['*.png', '*.webp', '*.tiff', '*.jpg', '*.jpeg', '*.bmp'] 
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    files = sorted(list(set(files)))
    
    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files)} images to process.")
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{i+1}/{len(files)}] Processing {filename}...")
        trim_image(file_path, output_path, padding, tolerance, quality)

def main():
    parser = argparse.ArgumentParser(description="Crop transparent borders or solid backgrounds from images.")
    parser.add_argument("input", help="Path to input image file OR directory of images")
    parser.add_argument("-o", "--output", required=True, help="Path to output image file OR output directory")
    parser.add_argument("-p", "--padding", type=int, default=20, help="Padding in pixels around the content (default: 20)")
    parser.add_argument("-t", "--tolerance", type=int, default=0, help="Tolerance for solid background color comparison (0-255). Recommended 5-10 for JPEGs. (default: 0)")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG Quality (1-100). Default: 95")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_batch(args.input, args.output, args.padding, args.tolerance, args.quality)
    else:
        trim_image(args.input, args.output, args.padding, args.tolerance, args.quality)
if __name__ == "__main__":
    main()
