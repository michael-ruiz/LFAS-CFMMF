#!/usr/bin/env python3
"""
WMCA HDF5 Image Extraction Script

Extracts individual modality images from WMCA HDF5 files for LFAS-CFMMF training.
Each HDF5 file contains 50 frames with 4 modalities (Color, Depth, IR, Thermal).

Based on WMCA documentation and Bob toolkit definitions:
- Type 0: Bonafide
- Type 1: Facial disguise (glasses, etc.)
- Type 2: Fake face (mannequins, masks)
- Type 3: Photo (prints)
- Type 4: Video (replay)

Author: Claude Code
Date: 2025-12-27
"""

import os
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json

# Try to import tqdm, fallback to simple iteration if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


class WMCAExtractor:
    def __init__(self, hdf5_root, output_root, sample_every_n=1):
        """
        Args:
            hdf5_root: Path to preprocessed-face-station_CDIT directory
            output_root: Path to output directory for extracted images
            sample_every_n: Extract every Nth frame (1 = all frames)
        """
        self.hdf5_root = Path(hdf5_root)
        self.output_root = Path(output_root)
        self.sample_every_n = sample_every_n

        # Create output directories for each modality
        self.modality_dirs = {}
        for modality in ['color', 'depth', 'ir', 'thermal']:
            modality_dir = self.output_root / modality
            modality_dir.mkdir(parents=True, exist_ok=True)
            self.modality_dirs[modality] = modality_dir

        # Statistics
        self.stats = {
            'total_files': 0,
            'total_frames_extracted': 0,
            'by_type_id': {},
            'errors': []
        }

    def parse_filename(self, filename):
        """Parse WMCA filename to extract metadata"""
        base = filename.replace('.hdf5', '')
        parts = base.split('_')

        if len(parts) != 5:
            return None

        return {
            'client_id': parts[0],
            'session_id': parts[1],
            'presenter_id': parts[2],
            'type_id': parts[3],
            'pai_id': parts[4],
            'label': 1 if parts[3] == '0' else 0,  # 1=bonafide, 0=attack
            'filename_base': base
        }

    def extract_frame(self, array_4ch, frame_idx, metadata, date_folder):
        """
        Extract and save individual modality images from 4-channel array

        Args:
            array_4ch: numpy array of shape (4, 128, 128)
            frame_idx: frame number
            metadata: parsed metadata from filename
            date_folder: date folder name for organization
        """
        # Verify shape
        if array_4ch.shape != (4, 128, 128):
            return False

        # Create output filename base
        output_base = f"{metadata['filename_base']}_frame{frame_idx:02d}.jpg"

        # Extract and save each modality
        modalities = ['color', 'depth', 'ir', 'thermal']
        for ch_idx, modality in enumerate(modalities):
            # Get channel data
            img = array_4ch[ch_idx]  # Shape: (128, 128)

            # Ensure uint8 format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Create date subfolder for organization
            output_dir = self.modality_dirs[modality] / date_folder
            output_dir.mkdir(exist_ok=True)

            # Save image using PIL
            output_path = output_dir / output_base
            pil_img = Image.fromarray(img)
            pil_img.save(str(output_path), quality=95)

        return True

    def process_hdf5_file(self, hdf5_path, date_folder):
        """Process a single HDF5 file"""
        filename = hdf5_path.name
        metadata = self.parse_filename(filename)

        if metadata is None:
            self.stats['errors'].append(f"Failed to parse filename: {filename}")
            return 0

        frames_extracted = 0

        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Get total number of frames
                frame_keys = [k for k in f.keys() if k.startswith('Frame_')]
                total_frames = len(frame_keys)

                # Extract frames
                for frame_idx in range(0, total_frames, self.sample_every_n):
                    frame_key = f'Frame_{frame_idx}'

                    if frame_key not in f:
                        continue

                    if 'array' not in f[frame_key]:
                        continue

                    array_4ch = f[frame_key]['array'][:]

                    if self.extract_frame(array_4ch, frame_idx, metadata, date_folder):
                        frames_extracted += 1

        except Exception as e:
            self.stats['errors'].append(f"Error processing {filename}: {str(e)}")
            return 0

        # Update statistics
        type_id = metadata['type_id']
        if type_id not in self.stats['by_type_id']:
            self.stats['by_type_id'][type_id] = {'files': 0, 'frames': 0}
        self.stats['by_type_id'][type_id]['files'] += 1
        self.stats['by_type_id'][type_id]['frames'] += frames_extracted

        return frames_extracted

    def extract_all(self, dry_run=False):
        """Extract images from all HDF5 files"""
        print(f"Starting WMCA image extraction")
        print(f"HDF5 root: {self.hdf5_root}")
        print(f"Output root: {self.output_root}")
        print(f"Sample every {self.sample_every_n} frame(s)")
        print("="*80)

        # Collect all HDF5 files
        hdf5_files = []
        for date_folder in sorted(self.hdf5_root.iterdir()):
            if not date_folder.is_dir():
                continue

            for hdf5_file in date_folder.glob('*.hdf5'):
                hdf5_files.append((hdf5_file, date_folder.name))

        print(f"Found {len(hdf5_files)} HDF5 files")

        if dry_run:
            print("\n[DRY RUN] Would process the following:")
            for hdf5_path, date_folder in hdf5_files[:10]:
                print(f"  {date_folder}/{hdf5_path.name}")
            print(f"  ... and {len(hdf5_files) - 10} more files")
            return

        # Process all files with progress bar
        print("\nExtracting images...")
        for hdf5_path, date_folder in tqdm(hdf5_files, desc="Processing files"):
            frames = self.process_hdf5_file(hdf5_path, date_folder)
            self.stats['total_files'] += 1
            self.stats['total_frames_extracted'] += frames

        # Print summary
        print("\n" + "="*80)
        print("Extraction Complete!")
        print("-"*80)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Total frames extracted: {self.stats['total_frames_extracted']}")
        print("\nBy Type ID:")

        type_names = {
            '0': 'Bonafide',
            '1': 'Facial disguise',
            '2': 'Fake face',
            '3': 'Photo/Print',
            '4': 'Video/Replay'
        }

        for type_id in sorted(self.stats['by_type_id'].keys()):
            stats = self.stats['by_type_id'][type_id]
            type_name = type_names.get(type_id, f'Unknown ({type_id})')
            print(f"  Type {type_id} ({type_name}): {stats['files']} files, {stats['frames']} frames")

        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:10]:
                print(f"  {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")

        # Save statistics
        stats_path = self.output_root / 'extraction_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")


def main():
    # Get the project root directory (parent of data_process folder)
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    datasets_dir = project_root / 'datasets'

    parser = argparse.ArgumentParser(description='Extract WMCA HDF5 images')
    parser.add_argument('--hdf5_root', type=str,
                       default=str(datasets_dir / 'WMCA' / 'WMCA' / 'preprocessed-face-station_CDIT'),
                       help='Path to HDF5 root directory')
    parser.add_argument('--output_root', type=str,
                       default=str(datasets_dir / 'WMCA-1' / 'preprocessed'),
                       help='Path to output directory')
    parser.add_argument('--sample_every_n', type=int, default=1,
                       help='Extract every Nth frame (1=all frames, 5=every 5th frame)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without actually extracting')

    args = parser.parse_args()

    extractor = WMCAExtractor(
        hdf5_root=args.hdf5_root,
        output_root=args.output_root,
        sample_every_n=args.sample_every_n
    )

    extractor.extract_all(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
