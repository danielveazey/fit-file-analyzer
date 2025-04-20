import fitparse
import csv
import argparse
import os
import sys
from datetime import datetime

def convert_fit_to_csv(fit_filepath, csv_filepath):
    """
    Converts a .FIT file to a .CSV file, including all data fields from all messages.

    Args:
        fit_filepath (str): Path to the input .FIT file.
        csv_filepath (str): Path to the output .CSV file.
    """
    if not os.path.exists(fit_filepath):
        print(f"Error: Input FIT file not found: {fit_filepath}")
        sys.exit(1)

    print(f"Processing FIT file: {fit_filepath}")

    try:
        fitfile_for_headers = fitparse.FitFile(fit_filepath)

        # --- Pass 1: Collect all unique field names across all message types ---
        print("Pass 1: Collecting all field names...")
        all_field_names = set()
        message_count = 0
        all_messages_data = [] # Store message data for second pass

        for message in fitfile_for_headers.get_messages():
            message_count += 1
            msg_data = message.get_values()
            all_messages_data.append({'name': message.name, 'data': msg_data}) # Store name and data
            # Get field names from the current message's data
            for field_name in msg_data.keys():
                all_field_names.add(field_name)

        if message_count == 0:
            print(f"Warning: No messages found in {fit_filepath}. Output CSV will be empty.")
            # Create an empty CSV file with just standard headers
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write minimal headers even if empty
                writer.writerow(['message_type', 'message_index_in_file'])
            print(f"Empty CSV file created: {csv_filepath}")
            return

        print(f"Found {len(all_field_names)} unique field names across {message_count} messages.")

        # Prepare header row for CSV: message type, message index, then sorted field names
        # Using 'message_index_in_file' to be explicit about its scope
        headers = ['message_type', 'message_index_in_file'] + sorted(list(all_field_names))

        # --- Pass 2: Write data to CSV using stored message data ---
        print(f"Pass 2: Writing data to CSV: {csv_filepath}")

        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers) # Write the header row

            # Iterate through the stored message data from Pass 1
            for index, msg_info in enumerate(all_messages_data):
                message_type = msg_info['name']
                message_data = msg_info['data']

                # Create a dictionary for the current row based on headers
                # Initialize with empty strings for all potential fields
                row_dict = {header: '' for header in headers}

                # Populate common fields
                row_dict['message_type'] = message_type
                # Global index of the message in the file (0-based from enumeration)
                row_dict['message_index_in_file'] = index

                # Populate specific fields found in this message's data
                for field_name, value in message_data.items():
                    # Check if the field name is actually in our collected headers
                    # (It should be, but defensive check doesn't hurt)
                    if field_name in row_dict:
                        # Optional: Format datetime objects for better readability in CSV
                        if isinstance(value, datetime):
                           # Use ISO 8601 format, standard and unambiguous
                           value = value.isoformat()
                        row_dict[field_name] = value

                # Write the row using the order defined by the headers list
                writer.writerow([row_dict[header] for header in headers])

        print(f"Successfully converted {fit_filepath} to {csv_filepath}")

    except fitparse.FitParseError as e:
        print(f"Error parsing FIT file '{fit_filepath}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during conversion of '{fit_filepath}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a GPS .FIT file to a .CSV file, including all data fields from all messages.")
    parser.add_argument("fit_file", help="Path to the input .FIT file.")
    parser.add_argument("csv_file", help="Path for the output .CSV file.")

    args = parser.parse_args()

    convert_fit_to_csv(args.fit_file, args.csv_file)
