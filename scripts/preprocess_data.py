import pandas as pd
import os  # Ensure directories exist

def preprocess_data(input_file, expected_file, output_file):

    print("DEBUG: Starting script...")
    print(f"Input file: {input_file}")
    print(f"Expected file: {expected_file}")
    print(f"Output file: {output_file}")
    
    # Load in.tsv (input file)
    input_data = pd.read_csv(input_file, sep='\t', quoting=3, header=None, names=['filename', 'keys', 'text'])
    
    # Load expected.tsv (expected output)
    expected_data = pd.read_csv(expected_file, sep='\t', quoting=3, header=None, names=['key_values'])

    # Parse key-value pairs from expected.tsv
    from collections import defaultdict
    def parse_key_values(row):
        key_value_pairs = row.split()
        label_dict = defaultdict(list)
        for pair in key_value_pairs:
            key, value = pair.split('=')
            label_dict[key].append(value)
        return {key: values[0] if len(values) == 1 else values for key, values in label_dict.items()}

    expected_data['labels'] = expected_data['key_values'].apply(parse_key_values)

    # Reset indices to ensure alignment
    input_data.reset_index(drop=True, inplace=True)
    expected_data.reset_index(drop=True, inplace=True)

    # Combine input text with parsed labels
    processed_data = pd.DataFrame({
        'text': input_data['text'],
        'labels': expected_data['labels']
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save preprocessed data
    processed_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    preprocess_data(
        input_file="data/raw/train/in.tsv",
        expected_file="data/raw/train/expected.tsv",
        output_file="data/processed/train_preprocessed.tsv"
    )