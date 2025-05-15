from modules.SignLanguageProcessor import SignLanguagePreprocessor
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sign Language Preprocessor')
    parser.add_argument('--raw', type=str, default="raw_data", help='Path dir of the raw dataset')
    parser.add_argument('--out', type=str, default="data", help='The output dir of preprocessed data')
    args = parser.parse_args()
    raw_data_dir = args.raw
    output_dir = args.out 

    preprocessor = SignLanguagePreprocessor(
        raw_data_dir=raw_data_dir,
        static_frames=3,
        output_dir=output_dir,
        min_confidence=0.8,
        min_hand_confidence=0.3,      
        force_process=True,
        debug_mode=True,              
        enhanced_detection=False       
    )

    preprocessor.process_all_data()

    print("Preprocessing complete!")