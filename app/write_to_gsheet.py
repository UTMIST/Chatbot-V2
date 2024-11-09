from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def write_to_gsheet(question: str, answer: str, description: str) -> None:
    # Set up the credentials (replace with your own credentials)
    creds = Credentials.from_authorized_user_file(
        "path/to/credentials.json", ["https://www.googleapis.com/auth/spreadsheets"]
    )

    # Set up the Google Sheets API service
    service = build("sheets", "v4", credentials=creds)

    # Set the spreadsheet IDs (replace with your own spreadsheet IDs)
    spreadsheet1_id = "your_spreadsheet1_id"
    spreadsheet2_id = "your_spreadsheet2_id"

    # Set the range for the first spreadsheet
    range1 = "Sheet1!A1:B1"

    # Set the values for the first spreadsheet
    values1 = [[question, answer]]

    # Write to the first spreadsheet
    body1 = {"values": values1}
    result1 = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet1_id,
            range=range1,
            valueInputOption="RAW",
            body=body1,
        )
        .execute()
    )

    # Set the range for the second spreadsheet
    range2 = "Sheet1!A1:A1"

    # Set the values for the second spreadsheet
    values2 = [[description]]

    # Write to the second spreadsheet
    body2 = {"values": values2}
    result2 = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet2_id,
            range=range2,
            valueInputOption="RAW",
            body=body2,
        )
        .execute()
    )

    print(
        f"Wrote to first spreadsheet: {result1.get('updates').get('updatedCells')} cells appended."
    )
    print(
        f"Wrote to second spreadsheet: {result2.get('updates').get('updatedCells')} cells appended."
    )
