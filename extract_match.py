import os
import re

# will extract the matching with different patterns possibilities exactly or similar to the expected output format
# from our experiences, when a matching is not found, it is either because: the name are not correct (not student and school) or there is no matching


def extract_first_matching_block(input_folder, output_folder):
    block_pattern = r"\[\s*((?:\(\s*[\"'‘’“”]?s\d+[\"'‘’“”]?\s*,\s*[\"'‘’“”]?(?:c\d+|nothing)[\"'‘’“”]?\s*\)\s*,?\s*)+)\]"
    line_tuple_pattern = r"\(\s*[\"'‘’“”]?(s\d+)[\"'‘’“”]?\s*,\s*[\"'‘’“”]?(c\d+|nothing)[\"'‘’“”]?\s*\)\s*,?"
    json_array_pattern = r'\[\s*"?(s\d+)"?\s*,\s*"?(c\d+|nothing)"?\s*\](?:,|\s|$)'
    escaped_tuple_pattern = r'\(\s*\\"(s\d+)\\"\s*,\s*\\"(c\d+|nothing)\\"\s*\)'


    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        matches = None

        block_matches = re.finditer(block_pattern, text, re.DOTALL)
        best_raw_match_list = None
        best_num_matches = 0

        for block in block_matches:
            raw_match_list = block.group(1)
            extracted_pairs = re.findall(line_tuple_pattern, raw_match_list)
            if len(extracted_pairs) > best_num_matches:
                best_num_matches = len(extracted_pairs)
                best_raw_match_list = raw_match_list

        if best_raw_match_list:
            matches = re.findall(line_tuple_pattern, best_raw_match_list)
        else:
            matches = None
        
        if not matches:
            lines = text.splitlines()
            match_lines = []
            started = False
            for line in lines:
                if re.search(line_tuple_pattern, line):
                    match_lines.append(line.strip())
                    started = True
                elif started:
                    break
            block_text = " ".join(match_lines)
            matches = re.findall(line_tuple_pattern, block_text)

        if not matches:
            matches = re.findall(escaped_tuple_pattern, text)
            
        if not matches:
            matches = re.findall(json_array_pattern, text)

        if not matches:
            print(f"No matching block found in: {filename}")
            continue

        matchings = [(s, c) for s, c in matches]
        output_lines = [f"({s}, {c})" for s, c in matchings]

        base_name = filename[:-4]  # Remove .txt
        output_filename = base_name + "_extract.txt"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract matching blocks from LLM-generated outputs."
    )
    parser.add_argument("--input", required=True, help="Path to input folder containing .txt files")
    parser.add_argument("--output", required=True, help="Path to output folder for extracted matchings")
    args = parser.parse_args()

    extract_first_matching_block(args.input, args.output)
