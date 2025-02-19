import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import os
from tqdm.auto import tqdm
import zipfile
import gc
from pathlib import Path

class WordExtractor:
    def __init__(self):
        self.categories = ['bold', 'italic', 'underlined', 'normal']
        self.output_dir = Path('word_dataset')
        self.word_count = {cat: 0 for cat in self.categories}
        self.dpi = 300  # Higher DPI for M4 chip
        # Set tesseract path for macOS
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
        self.create_folders()

    def create_folders(self):
        if self.output_dir.exists():
            print("Cleaning previous dataset...")
            for category in self.categories:
                folder_path = self.output_dir / category
                if folder_path.exists():
                    for file in folder_path.glob('*.png'):
                        file.unlink()
        
        for category in self.categories:
            (self.output_dir / category).mkdir(parents=True, exist_ok=True)
        print("Created directories")

    def detect_underline(self, img):
        """Improved underline detection"""
        height, width = img.shape
        bottom_region = img[int(height*0.7):, :]
        
        # Apply horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//3, 1))
        detected = cv2.morphologyEx(bottom_region, cv2.MORPH_OPEN, kernel)
        
        # Check for horizontal lines
        contours, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > width * 0.5 and h <= 2:  # Line should be at least half the width
                return True
        return False

    def enhance_underline(self, img):
        """Better underline enhancement"""
        height, width = img.shape
        enhanced = img.copy()
        
        # Add line at 85% of height
        line_y = int(height * 0.85)
        line_thickness = max(1, height // 30)  # Proportional thickness
        cv2.line(enhanced, (0, line_y), (width, line_y), 255, line_thickness)
        
        return enhanced

    def process_word_image(self, word_image, is_underlined=False):
        """Improved word image processing"""
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            word_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Clean noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        if is_underlined:
            if not self.detect_underline(binary):
                binary = self.enhance_underline(binary)
                
        return binary

    def process_page(self, image, page_num):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Improve OCR accuracy
        config = '--oem 1 --psm 11'
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        
        category = self.get_category(page_num)
        is_underlined = (category == 'underlined')
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                text = data['text'][i].strip()
                if not text:
                    continue
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                if w < 10 or h < 10:  # Minimum size threshold
                    continue
                
                # Extract with padding
                pad = 5
                word_image = gray[
                    max(0, y-pad):min(gray.shape[0], y+h+pad),
                    max(0, x-pad):min(gray.shape[1], x+w+pad)
                ]
                
                if word_image.size > 0:
                    processed_image = self.process_word_image(word_image, is_underlined)
                    
                    filename = self.output_dir / category / f"word_{self.word_count[category]}.png"
                    cv2.imwrite(str(filename), processed_image)
                    self.word_count[category] += 1
                    
                    if sum(self.word_count.values()) % 100 == 0:
                        print(f"Processed {sum(self.word_count.values())} words...")
        
        return self.word_count[category]

    def get_category(self, page_num):
        if page_num < 200:
            return 'bold'
        elif page_num < 400:
            return 'italic'
        elif page_num < 600:
            return 'underlined'
        return 'normal'

    def process_pdf(self, pdf_path):
        print("Starting PDF processing...")
        print("Converting PDF to images...")
        
        pages = convert_from_path(
            pdf_path, 
            dpi=self.dpi,
            thread_count=8,  # Optimized for M4
            poppler_path="/opt/homebrew/bin"  # macOS poppler path
        )
        total_pages = len(pages)
        print(f"Found {total_pages} pages")
        
        for page_num, page in tqdm(enumerate(pages), total=total_pages, desc="Processing pages"):
            words_in_page = self.process_page(page, page_num)
            print(f"Page {page_num + 1}/{total_pages} - Extracted {words_in_page} words")
            gc.collect()
        
        print("\nWords extracted per category:")
        for category, count in self.word_count.items():
            print(f"{category}: {count}")
        
        return sum(self.word_count.values())

    def create_zip(self):
        print("\nCreating ZIP archive...")
        zip_path = 'word_images_dataset.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            files = list(self.output_dir.rglob('*.png'))
            for file in tqdm(files, desc="Adding to ZIP"):
                zipf.write(file, file.relative_to(self.output_dir))
        
        print(f"Dataset saved to: {zip_path}")

def main():
    pdf_path = "/Users/karthikkurra/Desktop/IIIT_Works/DatasetAttrbute/final.pdf"
    print(f"Processing PDF: {pdf_path}")
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    extractor = WordExtractor()
    total_words = extractor.process_pdf(pdf_path)
    extractor.create_zip()
    print(f"\nProcess completed! Total words extracted: {total_words}")

if __name__ == "__main__":
    main()