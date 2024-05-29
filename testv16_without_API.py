import streamlit as st
from streamlit_multipage import MultiPage
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
import pandas as pd
import boto3
import botocore
import os
import anthropic
from datetime import datetime, timedelta
import json
import numpy as np
import re
import json

# Set AWS credentials and region
os.environ["AWS_ACCESS_KEY_ID"] = "API"
os.environ["AWS_SECRET_ACCESS_KEY"] = "API"
os.environ["REGION_NAME"] = "Region"

# Setup AWS session for S3
boto3.setup_default_session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            region_name=os.getenv('REGION_NAME'))

ANTHROPIC_API_KEY = "API"
NEWSAPI_KEY = "API"
CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-sonnet-20240229"

class AWSOperations:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def fetch_object(self, file_name, bucket_name):
        obj = self.s3.get_object(Bucket=bucket_name, Key=file_name)
        return obj['Body'].read().decode('utf-8')

class AIResponseGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_response(self, prompt, system_prompt, partner_letters, fund_names_dates):
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Create XML tags for each document
        tagged_letters = []
        for letter, fund_name_date in zip(partner_letters, fund_names_dates):
            fund_name, date = fund_name_date.lower().split(" ", 1)
            fund_name = fund_name.replace(" ", "")
            quarter = date.split(" ")[0].lower()
            year = date.split(" ")[1]
            tag = f"<{fund_name}_{year}_{quarter}>"
            tagged_letter = f"{tag}\n{letter}\n</{fund_name}_{year}_{quarter}>"
            tagged_letters.append(tagged_letter)
        
        combined_letters = "\n\n".join(tagged_letters)
        
        message = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{combined_letters}\n\n{prompt}"
                        }
                    ]
                }
            ]
        )
        
        raw_text = message.content
        answer = raw_text[0].text
        
        # Extract the content within <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        st.write(answer)

class DocumentFetcher:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_partner_letters(self, fund_names_dates):
        partner_letters = []
        for fund_name_date in fund_names_dates:
            parts = fund_name_date.split()
            fund_name = " ".join(parts[:-2]).lower().replace(" ", "")
            date = " ".join(parts[-2:])
            file_name = f"{fund_name}/cleaned/{fund_name_date}.txt"
            try:
                letter_content = self.aws_operations.fetch_object(file_name, "hedgefunds")
                partner_letters.append(letter_content)
            except Exception as e:
                print(f"File not found: {file_name}")
                print(f"Error: {str(e)}")
        return partner_letters

def fetch_fund_names(aws_operations, bucket_name, fund_info_path):
    # Fetch the JSON file from S3
    json_data = aws_operations.fetch_object(fund_info_path, bucket_name)
    fund_info_data = json.loads(json_data)

    # Get unique fund names
    fund_names = set(obj['Fund Name'].replace(", LP", "") for obj in fund_info_data)

    return list(fund_names)

class OpportunityScout:
    def __init__(self, aws_operations, bucket_name):
        self.aws_operations = aws_operations
        self.bucket_name = bucket_name

    def fetch_json_data(self, formatted_fund_name):
        json_file_path = f"{formatted_fund_name}/{formatted_fund_name}_equities.json"

        try:
            json_data = self.aws_operations.fetch_object(json_file_path, self.bucket_name)
            json_data = json.loads(json_data)
            return json_data
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"JSON file not found for fund: {formatted_fund_name}")
                return None
            else:
                raise e

    def filter_companies(self, json_data, sectors, start_quarter, end_quarter, position_status, position_type):
        filtered_companies = []

        for company in json_data:
            if sectors and company['Sector'] not in sectors:
                continue

            pitch_quarter = company['Date']

            if start_quarter and end_quarter:
                if pitch_quarter < start_quarter or pitch_quarter > end_quarter:
                    continue

            if position_status == "Position Added" and str(company['PositionOpen']) == '0':
                continue
            if position_status == "Position Exited" and str(company['PositionClose']) == '0':
                continue

            if position_type != "Both" and company['PositionType'] != position_type:
                continue

            filtered_companies.append(company)

        return filtered_companies

    def aggregate_companies(self, selected_funds, sectors, start_quarter, end_quarter, position_status, position_type):
        aggregated_companies = []

        for fund_name in selected_funds:
            json_data = self.fetch_json_data(fund_name)
            filtered_companies = self.filter_companies(json_data, sectors, start_quarter, end_quarter, position_status, position_type)
            aggregated_companies.extend(filtered_companies)

        return aggregated_companies

    def display_companies(self, aggregated_companies):
        if not aggregated_companies:
            st.write("No companies found matching the selected criteria.")
            return

        df = pd.DataFrame(aggregated_companies)
        
        # Rename the columns
        df = df.rename(columns={
            "PositionType": "Position Type",
            "PositionOpen": "Position Added",
            "PositionClose": "Position Exited"
        })
        
        # Reorder the columns
        column_order = ["Fund", "Date", "Company", "Ticker", "Sector", "Thesis", "Position Type", "Position Added", "Position Exited"]
        df = df[column_order]
        
        # Add the message to inform the user about double-clicking on a cell
        st.write("**Double-click on a cell to see the full text.**")
        
        st.dataframe(df)

    def get_top_sectors(self, selected_funds, start_quarter, end_quarter):
        sector_counts = {}

        for fund_name in selected_funds:
            json_data = self.fetch_json_data(fund_name)
            filtered_companies = self.filter_companies(json_data, None, start_quarter, end_quarter, "Both", "Both")

            for company in filtered_companies:
                sector = company['Sector']
                if sector in sector_counts:
                    sector_counts[sector] += 1
                else:
                    sector_counts[sector] = 1

        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        top_sectors = sorted_sectors[:3]

        return top_sectors
    
    def run(self, fund_type, selected_funds):
        if not selected_funds:
            st.write("**Please Select At Least One Fund In The Side Bar**")
            return

        quarters = ["2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1"]
        start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=quarters, value=(quarters[0], quarters[-1]))

        if selected_funds:
            top_sectors = self.get_top_sectors(selected_funds, start_quarter, end_quarter)
            if top_sectors:
                st.write(f"**Company Sectors Most Frequently Discussed By The Selected Funds From {start_quarter} - {end_quarter}:**")
                for sector, count in top_sectors:
                    st.write(f"- {sector} ({count} mentions)")
            else:
                st.write(f"No sectors discussed by the selected funds from {start_quarter} to {end_quarter}.")

        sectors = ["All", "Financials", "Energy", "Health Care", "Communication Services", "Industrials", "Information Technology", "Consumer Discretionary", "Real Estate"]
        st.write("**Add Filters Below:**")
        selected_sectors = st.multiselect("Select Sectors", sectors)

        # Handle the "All" option
        if "All" in selected_sectors:
            selected_sectors = sectors[1:]  # Include all sectors except "All"

        position_statuses = ["Both", "Position Added", "Position Exited"]
        selected_position_status = st.radio("Select Security Position Status", position_statuses)

        position_types = ["Both", "Long", "Short"]
        selected_position_type = st.radio("Select Security Position Type", position_types)

        if st.button("Submit"):
            aggregated_companies = self.aggregate_companies(selected_funds, selected_sectors, start_quarter, end_quarter, selected_position_status, selected_position_type)
            self.display_companies(aggregated_companies)
                 
class PerformancePulse:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_performance_data(self, selected_funds, selected_quarter):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_data = json.loads(performance_data)

        # Filter the performance data based on the selected funds and quarter
        filtered_data = [
            obj for obj in performance_data
            if obj['Fund Name'] in selected_funds and obj['Date'] == selected_quarter
        ]

        return filtered_data

    def display_performance_table(self, filtered_data):
        # Extract the relevant columns from the filtered data for the performance table
        table_data = [
            {
                'Fund Name': obj['Fund Name'],
                'Quarterly Net Performance': self.convert_to_percentage(obj.get('Quarterly Performance Net of Fees', '')),
                'YTD Net Performance': self.convert_to_percentage(obj.get('Year-to-Date Performance Net of Fees', '')),
                'ITD Annualized Net Performance': self.convert_to_percentage(obj.get('Inception-to-Date Annualized Performance Net of Fees', ''))
            }
            for obj in filtered_data
        ]

        # Create a DataFrame from the table data
        df = pd.DataFrame(table_data)

        st.table(df)

    def convert_to_percentage(self, value):
        # Convert a value to percentage format
        try:
            return f"{float(value):.1f}%"
        except (ValueError, TypeError):
            return value

    def run(self, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund to view performance data.**")
            return

        # Fetch the performance insights data from the JSON file in the S3 bucket
        performance_insights_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_insights_data = json.loads(performance_insights_data)

        # Extract unique quarters from the performance insights data
        quarters = list(set(obj['Date'] for obj in performance_insights_data))
        quarters.sort(reverse=True)

        selected_quarter = st.selectbox("Select a quarter", options=quarters)

        if selected_funds:
            # Check if "All" is selected
            if "All" in selected_funds:
                # Get all unique fund names from the performance insights data
                selected_funds = list(set(obj['Fund Name'] for obj in performance_insights_data))

            filtered_data = self.fetch_performance_data(selected_funds, selected_quarter)

            if filtered_data:
                self.display_performance_table(filtered_data)

                # Add radio options for commentary selection
                commentary_options = ["Investment Landscape", "Portfolio Positioning", "Both"]
                selected_commentary = st.radio("Choose Commentary:", options=commentary_options)

                # Add a "Submit" button
                if st.button("Submit"):
                    # Display the selected commentary for each fund
                    for fund in selected_funds:
                        fund_data = next((obj for obj in performance_insights_data if obj['Fund Name'] == fund and obj['Date'] == selected_quarter), None)

                        if fund_data:
                            st.markdown(f"**{fund}**")

                            if selected_commentary in ["Investment Landscape", "Both"]:
                                st.markdown("<span style='color: #6E7C8C;'><strong>Commentary on the Investment Landscape:</strong></span>", unsafe_allow_html=True)
                                investment_landscape = fund_data.get("Investment Landscape", "")
                                if investment_landscape:
                                    st.write(investment_landscape)
                                else:
                                    st.write("No commentary available for Investment Landscape.")

                            if selected_commentary in ["Portfolio Positioning", "Both"]:
                                st.markdown("<span style='color: #6E7C8C;'><strong>Commentary on the Portfolio Positioning:</strong></span>", unsafe_allow_html=True)
                                portfolio_positioning = fund_data.get("Portfolio Positioning", "")
                                if portfolio_positioning:
                                    st.write(portfolio_positioning)
                                else:
                                    st.write("No commentary available for Portfolio Positioning.")

                            st.write("---")
            else:
                st.write("There Is No Performance Data On The Selected Fund and Date")
        else:
            st.write("**Please select at least one fund to view performance data.**")

class MarketMoodMonitor:
    def __init__(self, aws_operations, ai_response_generator, fund_info_path, document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.fund_info_path = fund_info_path
        self.document_fetcher = document_fetcher

    def fetch_fund_info_data(self):
        fund_info_data = self.aws_operations.fetch_object(self.fund_info_path, "hedgefunds")
        fund_info_data = json.loads(fund_info_data)
        return fund_info_data
    
    def get_unique_values(self, fund_info_data, key):
        values = set()
        for obj in fund_info_data:
            if key in obj:
                values.update(obj[key].split(", "))
        return list(values)

    def handle_theme_specific(self, fund_info_data, analysis_type, selected_funds, start_quarter, end_quarter):
        if analysis_type == 'Market Commentary':
            themes = self.get_unique_values(fund_info_data, 'Macro')
        elif analysis_type == 'Asset Class':
            themes = self.get_unique_values(fund_info_data, 'Asset Classes')
        elif analysis_type == 'Geography':
            themes = self.get_unique_values(fund_info_data, 'Geographies')

        selected_themes = st.multiselect(f'Select {analysis_type.lower()} themes:', themes)

        if selected_themes:
            # Add text to inform users about uploading documents
            st.write("**You can upload a maximum of 5 documents to filter in the sidebar.**")

            # Filter the fund_info_data based on the selected themes
            filtered_funds_data = []
            for obj in fund_info_data:
                if analysis_type == 'Market Commentary' and any(theme in obj.get('Macro', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)
                elif analysis_type == 'Asset Class' and any(theme in obj.get('Asset Classes', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)
                elif analysis_type == 'Geography' and any(theme in obj.get('Geographies', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)

            # Filter the filtered_funds_data based on the selected date range
            filtered_funds_data = [obj for obj in filtered_funds_data if start_quarter <= obj['Date'] <= end_quarter]

            # Filter the filtered_funds_data based on the selected funds
            if selected_funds:
                filtered_funds_data = [obj for obj in filtered_funds_data if obj['Fund Name'] in selected_funds]

            if filtered_funds_data:
                # Get the fund names and dates from the filtered data
                fund_names_dates = [f"{obj['Fund Name']} {obj['Date']}" for obj in filtered_funds_data]

                st.write(f"Funds and quarters that discussed the selected {analysis_type.lower()} themes:")
                for fund_name_date in fund_names_dates:
                    st.write(f"- {fund_name_date}")

                if st.button('Submit'):
                    message_prompt = f"Provide a detailed overview of the {', '.join(selected_themes)} themes discussed in the selected partner letters. Break it down into clear, structured bullet points. Highlight the key points that the letters discussed. Please compare and contrast between the different funds and quarters. Make sure to cite your sources by putting the title of the letter that was cited in brackets like this: [Greenlight Capital 2023 Q4]. I want to know the exactly source of each view point. The purpose is to make the user aware of the outlook on these specific themes. Please present your findings in a well-structured, easy-to-follow format."
                    
                    system_prompt = f"You are an experienced investment analyst with a deep understanding of various {analysis_type.lower()} themes. I have attached partner letters from the selected hedge funds for you to analyze and reference for the upcoming task. Each letter is identified by the appropiate XML tags at the top and bottom of the letter. Each hedge fund writes a quarterly partner letter discussing topics such as their performance, \
                        macroeconomic views, and rationale for adding specific equity positions to their fund. Please carefully read through the entire document and identify the most relevant commentary related to the selected themes: {', '.join(selected_themes)}. When you complete your task, first plan how you should answer and which data you will use within \
                            <thinking> </thinking> XML tags. This is a space for you to write down relevant content and will not be shown to the user. Once you are done thinking, output your final answer to the user within <answer> </answer> XML tags. Do not include closing tags or unnecessary open-and-close tag sections."

                    partner_letters = self.document_fetcher.fetch_partner_letters(fund_names_dates)
                    
                    # Display the included document names
                    st.write("These funds were included in the analysis:")
                    for fund_name_date in fund_names_dates:
                        st.write(f"- {fund_name_date}")

                    # Generate the response
                    self.ai_response_generator.generate_response(message_prompt, system_prompt, partner_letters, fund_names_dates)
            else:
                st.write(f"No funds found that discussed the selected {analysis_type.lower()} themes.")
    
    def run(self, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund to analyze.**")
            return

        analysis_type = st.radio('Select analysis type:', ['Market Commentary', 'Asset Class', 'Geography'])
        fund_info_data = self.fetch_fund_info_data()
            
        # Add a slider for selecting the date range
        quarters = ["2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1"]
        start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=quarters, value=(quarters[0], quarters[-1]))

        # Check if "All" is selected
        if "All" in selected_funds:
            # Get all unique fund names from the fund_info_data
            selected_funds = list(set(obj['Fund Name'] for obj in fund_info_data))

        self.handle_theme_specific(fund_info_data, analysis_type, selected_funds, start_quarter, end_quarter)

class MediaAndEvents:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_firm_updates_data(self):
        firm_updates_data = self.aws_operations.fetch_object("hedgefund_firm_updates.json", "hedgefunds")
        firm_updates_data = json.loads(firm_updates_data)
        return firm_updates_data
    
    def run(self, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund to view media and events updates.**")
            return

        firm_updates_data = self.fetch_firm_updates_data()
        unique_dates = sorted(set(obj['Date'] for obj in firm_updates_data), reverse=True)

        selected_date = st.selectbox("Select a date", unique_dates)
        update_type = st.radio("Select update type", ("Media Update", "Event Update"))

        if st.button("Submit"):
            # Check if "All" is selected
            if "All" in selected_funds:
                # Get all unique fund names from the firm updates data
                selected_funds = list(set(obj['Fund Name'] for obj in firm_updates_data))

            filtered_data = [obj for obj in firm_updates_data if obj['Fund Name'] in selected_funds and obj['Date'] == selected_date]

            for fund_data in filtered_data:
                fund_name = fund_data['Fund Name']
                update_content = fund_data.get(update_type, "")

                if update_content:
                    st.markdown(f"**{fund_name}**")
                    st.markdown(f"<span style='color: #6E7C8C;'><strong>{update_type}</strong></span>", unsafe_allow_html=True)
                    st.write(update_content)
                else:
                    st.write(f"<span style='color: #6E7C8C;'>{fund_name} does not have any {update_type} updates for the {selected_date} quarter.</span>", unsafe_allow_html=True)

                st.write("---")

    # def run(self, selected_funds):
    #     firm_updates_data = self.fetch_firm_updates_data()
    #     unique_dates = sorted(set(obj['Date'] for obj in firm_updates_data), reverse=True)

    #     selected_date = st.selectbox("Select a date", unique_dates)

    #     update_type = st.radio("Select update type", ("Media Update", "Event Update"))

    #     filtered_data = [obj for obj in firm_updates_data if obj['Fund Name'] in selected_funds and obj['Date'] == selected_date]

    #     for fund_data in filtered_data:
    #         fund_name = fund_data['Fund Name']
    #         update_content = fund_data.get(update_type, "")

    #         if update_content:
    #             st.markdown(f"**{fund_name}**")
    #             st.markdown(f"<span style='color: #6E7C8C;'><strong>{update_type}</strong></span>", unsafe_allow_html=True)
    #             st.write(update_content)
    #         else:
    #             st.write(f"{fund_name} does not have any {update_type} updates for the {selected_date} quarter.")

    #         st.write("---")

class SpecificFundsSection:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_available_quarters(self, selected_fund):
        # Fetch the JSON file containing the fund information
        fund_info_data = self.aws_operations.fetch_object("hedgefund_general_insights.json", "hedgefunds")
        fund_info_data = json.loads(fund_info_data)

        # Extract the available quarters for the selected fund
        available_quarters = [obj['Date'] for obj in fund_info_data if obj['Fund Name'] == selected_fund]

        # Sort the quarters based on year and quarter number
        def sort_key(quarter):
            year, quarter_num = quarter.split(' ')
            return (int(year), int(quarter_num[1]))

        available_quarters.sort(key=sort_key, reverse=True)

        return available_quarters

    def fetch_markdown_file(self, selected_fund, selected_quarter):
        # Format the markdown file name
        markdown_file_name = f"{selected_fund.lower().replace(' ', '')}/cleaned/sum_med {selected_fund} {selected_quarter}.md"

        try:
            # Fetch the markdown file from S3
            markdown_content = self.aws_operations.fetch_object(markdown_file_name, "hedgefunds")
            return markdown_content
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"Markdown file not found for fund: {selected_fund} and quarter: {selected_quarter}")
                return None
            else:
                raise e

    def fetch_pdf_file(self, selected_fund, selected_quarter):
        # Format the PDF file name
        pdf_file_name = f"{selected_fund}/{selected_fund} {selected_quarter} Summary.pdf"

        try:
            # Fetch the PDF file from S3
            pdf_content = self.aws_operations.fetch_object(pdf_file_name, "hedgefunds")
            return pdf_content
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"PDF file not found for fund: {selected_fund} and quarter: {selected_quarter}")
                return None
            else:
                raise e
            
    def fetch_anomalies_data(self, selected_fund, selected_quarter):
        anomalies_data = self.aws_operations.fetch_object("hedgefund_anomalies.json", "hedgefunds")
        anomalies_data = json.loads(anomalies_data)
        
        filtered_data = [obj for obj in anomalies_data if obj['Fund Name'] == selected_fund and obj['Date'] == selected_quarter]
        
        return filtered_data

    def fetch_firm_updates_data(self, selected_fund, selected_quarter):
        firm_updates_data = self.aws_operations.fetch_object("hedgefund_firm_updates.json", "hedgefunds")
        firm_updates_data = json.loads(firm_updates_data)
        
        filtered_data = [obj for obj in firm_updates_data if obj['Fund Name'] == selected_fund and obj['Date'] == selected_quarter]
        
        return filtered_data
    def fetch_performance_data(self, selected_fund):
        performance_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_data = json.loads(performance_data)
        
        filtered_data = [obj for obj in performance_data if obj['Fund Name'] == selected_fund]
        
        return filtered_data
    
    def run(self, selected_fund):
        if selected_fund:
            available_quarters = self.fetch_available_quarters(selected_fund)

            section_options = ["Summary", "Performance", "Notable Anomalies", "Firm Updates & Events", "Ask Anything"]
            selected_section = st.sidebar.radio("Select Analysis", section_options)

            if selected_section != "Performance":
                selected_quarter = st.sidebar.selectbox("Select Quarter", available_quarters, index=0)

            if selected_section == "Summary":
                if "selected_quarter" not in locals():
                    st.write("Please select a quarter.")
                else:
                    markdown_content = self.fetch_markdown_file(selected_fund, selected_quarter)
                    if markdown_content:
                        st.markdown(markdown_content)

                        # Offer the PDF for download when the user clicks the download button
                        if st.button("Download Summary as PDF"):
                            pdf_content = self.fetch_pdf_file(selected_fund, selected_quarter)
                            if pdf_content:
                                st.download_button(
                                    label="Download",
                                    data=pdf_content,
                                    file_name=f"{selected_fund} {selected_quarter} Summary.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.write("PDF file not available for download.")
                    else:
                        st.write("No summary available for the selected fund and quarter.")
            elif selected_section == "Performance":
                st.title(selected_fund)
                performance_data = self.fetch_performance_data(selected_fund)
                
                if performance_data:
                    quarters = sorted(set(obj['Date'] for obj in performance_data))
                    
                    start_quarter, end_quarter = st.sidebar.select_slider(
                        "Select Date Range",
                        options=quarters,
                        value=(quarters[0], quarters[-1])
                    )
                    
                    filtered_data = [obj for obj in performance_data if start_quarter <= obj['Date'] <= end_quarter]
                    
                    if filtered_data:
                        table_data = []
                        for obj in filtered_data:
                            table_data.append({
                                'Date': obj['Date'],
                                'Quarterly Net Performance': obj.get('Quarterly Performance Net of Fees', ''),
                                'YTD Net Performance': obj.get('Year-to-Date Performance Net of Fees', ''),
                                'ITD Annualized Net Performance': obj.get('Inception-to-Date Annualized Performance Net of Fees', '')
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.table(df)
                        
                        attribution_options = ["Attribution", "Macro+Positioning", "Both"]
                        selected_attribution = st.radio("Select Commentary", attribution_options)
                        
                        if st.button("Submit"):
                            for obj in filtered_data:
                                quarter = obj['Date']
                                st.markdown(f"**{quarter}**")
                                
                                if selected_attribution == "Attribution" or selected_attribution == "Both":
                                    key_contributors = obj.get('Key Contributors to Performance', '')
                                    if key_contributors:
                                        st.markdown("<span style='color: #6E7C8C;'><strong>Key Contributors to Performance:</strong></span>", unsafe_allow_html=True)
                                        st.write(key_contributors)
                                    
                                    key_detractors = obj.get('Key Detractors from Performance', '')
                                    if key_detractors:
                                        st.markdown("<span style='color: #6E7C8C;'><strong>Key Detractors from Performance:</strong></span>", unsafe_allow_html=True)
                                        st.write(key_detractors)
                                
                                if selected_attribution == "Macro+Positioning" or selected_attribution == "Both":
                                    investment_landscape = obj.get('Investment Landscape', '')
                                    if investment_landscape:
                                        st.markdown("<span style='color: #6E7C8C;'><strong>Investment Landscape:</strong></span>", unsafe_allow_html=True)
                                        st.write(investment_landscape)
                                    
                                    portfolio_positioning = obj.get('Portfolio Positioning', '')
                                    if portfolio_positioning:
                                        st.markdown("<span style='color: #6E7C8C;'><strong>Portfolio Positioning:</strong></span>", unsafe_allow_html=True)
                                        st.write(portfolio_positioning)
                                
                                st.write("---")
                    else:
                        st.write("No performance data available for the selected date range.")
                else:
                    st.write("No performance data available for the selected fund.")
            elif selected_section == "Notable Anomalies":
                st.title(selected_fund)
                anomalies_data = self.fetch_anomalies_data(selected_fund, selected_quarter)
                
                if anomalies_data:
                    st.subheader(f"**{selected_quarter}**")
                    
                    notable_anomalies = anomalies_data[0].get("Notable Anomalies", "")
                    st.markdown(f"<span style='color: #6E7C8C;'><strong>Notable Anomalies (using the previous 2 quarters for more context):</strong></span>", unsafe_allow_html=True)
                    st.write(notable_anomalies)
                else:
                    st.write("No notable anomalies found for the selected fund and quarter.")
            elif selected_section == "Firm Updates & Events":
                st.title(selected_fund)
                firm_updates_data = self.fetch_firm_updates_data(selected_fund, selected_quarter)
                
                if firm_updates_data:
                    st.subheader(f"**{selected_quarter}**")
                    
                    for key in ["Business Update", "Employee Update", "Media Update", "Event Update", "Additional Business Updates"]:
                        value = firm_updates_data[0].get(key, "")
                        if value:
                            st.markdown(f"<span style='color: #6E7C8C;'><strong>{key}:</strong></span>", unsafe_allow_html=True)
                            st.write(value)
                        else:
                            pass
                else:
                    st.write("No firm updates and events found for the selected fund and quarter.")
            elif selected_section == "Ask Anything":
                st.title(selected_fund)
                st.write("Ask Anything section will be implemented later.")

class VCDocumentFetcher:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_vc_partner_letters(self, fund_name, date):
        bucket_name = fund_name.split(" ")[0].lower()
        file_name = f"{bucket_name}/cleaned/{fund_name} {date}.txt"
        try:
            letter_content = self.aws_operations.fetch_object(file_name, "venturecapitalfunds")
            return letter_content
        except Exception as e:
            print(f"File not found: {file_name}")
            print(f"Error: {str(e)}")
            return None

class VCOpportunityScout:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_investments_data(self, selected_funds):
        investments_data = []
        for fund_name in selected_funds:
            bucket_name = fund_name.split(" ")[0].lower()
            file_name = f"{bucket_name}/{bucket_name}_investments.json"
            try:
                json_data = self.aws_operations.fetch_object(file_name, "venturecapitalfunds")
                investments_data.extend(json.loads(json_data))
            except Exception as e:
                print(f"File not found: {file_name}")
                print(f"Error: {str(e)}")
        return investments_data
    
    def run(self, fund_type, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund in the side bar**")
            return
        
        if fund_type == "Venture Capital Funds":
            investments_data = self.fetch_investments_data(selected_funds)
            investment_types = sorted(set(item["Type of Investment"] for item in investments_data))
            selected_investment_type = st.selectbox("Select Type of Investment", ["All"] + investment_types)

            # Add the "Select Amount Invested" filter
            amount_invested_options = ["All", "<$1m", "$1m-$10m", ">$10m"]
            selected_amount_invested = st.selectbox("Select Amount Invested", amount_invested_options)

            # Add the "Fair Value of the Investment" filter
            fair_value_options = sorted(set(item["Fair Value of the Investment"] for item in investments_data))
            selected_fair_value = st.selectbox("Select Fair Value of the Investment", ["All"] + fair_value_options)

            if st.button("Submit"):
                filtered_data = investments_data

                if selected_investment_type != "All":
                    filtered_data = [item for item in filtered_data if item["Type of Investment"] == selected_investment_type]

                # Apply the "Select Amount Invested" filter
                if selected_amount_invested != "All":
                    if selected_amount_invested == "<$1m":
                        filtered_data = [item for item in filtered_data if float(item["Amount Invested"]) < 1]
                    elif selected_amount_invested == "$1m-$10m":
                        filtered_data = [item for item in filtered_data if 1 <= float(item["Amount Invested"]) <= 10]
                    else:  # ">$10m"
                        filtered_data = [item for item in filtered_data if float(item["Amount Invested"]) > 10]

                # Apply the "Fair Value of the Investment" filter
                if selected_fair_value != "All":
                    filtered_data = [item for item in filtered_data if item["Fair Value of the Investment"] == selected_fair_value]

                if filtered_data:
                    df = pd.DataFrame(filtered_data)
                    df["Amount Invested"] = df["Amount Invested"].apply(lambda x: "${:,.2f}".format(float(x)))
                    df = df[["Fund", "Date", "Company", "Type of Investment", "Amount Invested", "Date invested", "Fair Value of the Investment", "Summary"]]
                    
                    df.reset_index(drop=True, inplace=True)

                    # Add the message to inform the user about double-clicking on a cell
                    st.write("**Double-click on a cell to see the full text.**")

                    st.dataframe(df)
                else:
                    st.write("No data found for the selected filters.")
        else:
            st.write("This feature is not available for the selected fund type.")  
            
class SpecificVCFundsSection:
    def __init__(self, aws_operations, ai_response_generator, vc_document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.vc_document_fetcher = vc_document_fetcher

    def fetch_performance_data(self, selected_fund):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("vc_performance_insights.json", "venturecapitalfunds")
        performance_data = json.loads(performance_data)

        # Filter the performance data based on the selected fund
        filtered_data = [obj for obj in performance_data if obj['Fund Name'] == selected_fund]

        return filtered_data

    def display_performance_table(self, filtered_data):
        # Extract the relevant columns from the filtered data for the performance table
        table_data = [
            {
                'Date': obj['Date'],
                'Net IRR': obj['Net IRR'],
                'Percentage Capital Commitments Called': obj['Percentage Capital Commitments Called']
            }
            for obj in filtered_data
        ]

        # Create a DataFrame from the table data
        df = pd.DataFrame(table_data)

        st.table(df)

    def display_selected_text(self, filtered_data, selected_option):
        # Extract the text based on the selected option
        text = filtered_data[0].get(selected_option, '')

        st.write(text)

    def generate_vc_response(self, prompt, system_prompt, partner_letter, fund_name, date):
        client = anthropic.Anthropic(api_key=self.ai_response_generator.api_key)
        
        # Create XML tags for the document
        fund_name = fund_name.replace(" ", "").replace(",", "")
        quarter = date.split(" ")[0].lower()
        year = date.split(" ")[1]
        tag = f"<{fund_name}_{year}_{quarter}>"
        tagged_letter = f"{tag}\n{partner_letter}\n</{fund_name}_{year}_{quarter}>"
        
        message = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{tagged_letter}\n\n{prompt}"
                        }
                    ]
                }
            ]
        )
        
        raw_text = message.content
        answer = raw_text[0].text
        
        # Extract the content within <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        st.write(answer)

    def handle_ask_anything(self, selected_fund, filtered_data):
        user_input = st.text_input("Enter your question:")

        if user_input:
            date = filtered_data[0]['Date']
            partner_letter = self.vc_document_fetcher.fetch_vc_partner_letters(selected_fund, date)

            if partner_letter:
                system_prompt = f"""
                You are an experienced venture capital analyst reviewing the quarterly letter from the VC fund {selected_fund}.
                Venture capitalists invest in early-stage, high-growth potential companies with the goal of generating significant returns for their investors. They write quarterly letters to provide updates on the fund's performance, portfolio companies, and market insights to their limited partners (investors).
                The user will ask a question about the provided quarterly letter and I want you to do the best job at answering it.
                Your task is to analyze the provided partner letter and extract relevant insights to answer the user's question.
                Provide a comprehensive response, citing specific examples from the letter to support your points.
                After reviewing the relevant information, take a moment to organize your thoughts and present your final analysis to the user.
                """

                message_prompt = f"""
                This is the user's question:
                <user_input>
                {user_input}
                </user_input>

                Please provide a detailed response based on the information in the quarterly letter from {selected_fund}. Make sure you cite your sources correctly and provide a well-structured answer.
                """

                self.generate_vc_response(message_prompt, system_prompt, partner_letter, selected_fund, date)
            else:
                st.write("Partner letter not found for the selected fund and date.")

    def run(self, selected_fund):
        if selected_fund:
            st.title(selected_fund)

            filtered_data = self.fetch_performance_data(selected_fund)

            if filtered_data:
                self.display_performance_table(filtered_data)

                options = [
                    "Commentary on Fund Performance",
                    "Key Contributors to Performance",
                    "Key Detractors from Performance",
                    "Portfolio Positioning and Adjustments",
                    "Ask Anything"
                ]
                selected_option = st.selectbox("Select an option", options)

                if selected_option == "Ask Anything":
                    self.handle_ask_anything(selected_fund, filtered_data)
                elif st.button("Submit"):
                    self.display_selected_text(filtered_data, selected_option)
            else:
                st.write("No performance data found for the selected fund.")

class SourcesSection:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_fund_names(self, bucket_name, fund_info_path):
        # Fetch the JSON file from S3
        json_data = self.aws_operations.fetch_object(fund_info_path, bucket_name)
        fund_info_data = json.loads(json_data)

        # Get unique fund names
        fund_names = set(obj['Fund Name'] for obj in fund_info_data)

        return list(fund_names)

    def run(self):
        source_option = st.radio(
            "Select a source option",
            ("Hedge Fund Partner Letters", "Podcasts", "VC Documents")
        )

        if source_option == "Hedge Fund Partner Letters":
            bucket_name = "hedgefunds"
            fund_info_path = "hedgefund_general_insights.json"
            fund_names = self.fetch_fund_names(bucket_name, fund_info_path)

            if len(fund_names) > 0:
                # Create a one-column DataFrame with fund names
                fund_data = pd.DataFrame({"Fund Name": fund_names})

                # Display the DataFrame
                st.write("Fund Names:")
                st.dataframe(fund_data)
            else:
                st.write("No fund names found.")

        elif source_option == "Podcasts":
            st.write("Implement the logic for 'Podcasts'")
            # Add your implementation here

        elif source_option == "VC Documents":
            st.write("Implement the logic for 'VC Documents'")
            # Add your implementation here

        # File upload section
        st.write("Upload your own documents:")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file to S3
            self.aws_operations.upload_object(uploaded_file.getvalue(), uploaded_file.name)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

def main():
    st.set_page_config(layout="wide")
    aws_operations = AWSOperations()
    ai_response_generator = AIResponseGenerator(ANTHROPIC_API_KEY)
    document_fetcher = DocumentFetcher(aws_operations)
    market_mood_monitor = MarketMoodMonitor(aws_operations, ai_response_generator, "hedgefund_general_insights.json", document_fetcher)
    sources_section = SourcesSection(aws_operations)
    performance_pulse = PerformancePulse(aws_operations)
    specific_funds_section = SpecificFundsSection(aws_operations)
    vc_document_fetcher = VCDocumentFetcher(aws_operations)
    specific_vc_funds_section = SpecificVCFundsSection(aws_operations, ai_response_generator, vc_document_fetcher)
    vc_opportunity_scout = VCOpportunityScout(aws_operations) 

    selected_option = st.sidebar.radio(
        "Navigation",
        ("Home", "Bird's-Eye View (Multiple Funds)", "Deep Dive (Single Fund)", "Sources")
    )

    if selected_option == "Home":
        st.markdown("<h1 style='text-align: center; color: blue;'>🚀 Welcome to wybe.ai!</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center'>
            <br>
            <p style='font-size: 24px;'>
                Asset allocators receive many of documents each month. These documents contain valuable information, but it can be challenging and time-consuming to locate specific insights.<br><br>
                This application leverages <b style='color: blue;'>Generative AI</b> to address the <b style='color: blue;'>information overload problem</b>. It automatically extracts and highlights key <b><span style='color: red;'>red</span></b> and <b><span style='color: green;'>green</span></b> flags from letters and shares this with asset allocators so that they can <b style='color: blue;'>identify critical insights quickly</b> in an easy to read curated manner.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif selected_option == "Bird's-Eye View (Multiple Funds)":
        fund_type = st.sidebar.radio(
            "Select a Fund Type",
            ("Hedge Funds", "Venture Capital Funds", "Private Equity Funds"),
            key="main_fund_type"
        )

        if fund_type == "Venture Capital Funds":
            asset_allocator_options = ["Opportunity Scout"]
        else:
            asset_allocator_options = ["Opportunity Scout", "Performance Pulse", "Market Mood Monitor", "Media and Events"]

        asset_allocator_option = st.sidebar.selectbox(
            "Select an option",
            asset_allocator_options
        )

        bucket_name = get_bucket_name(fund_type)

        if bucket_name:
            if fund_type == "Hedge Funds":
                fund_insights_path = "hedgefund_general_insights.json"
            elif fund_type == "Venture Capital Funds":
                fund_insights_path = "vc_performance_insights.json"
            else:
                fund_insights_path = None

            if fund_insights_path:
                selected_funds = select_funds(aws_operations, bucket_name, fund_insights_path)
                formatted_selected_funds = format_fund_names(selected_funds, fund_type)
            else:
                formatted_selected_funds = []
        else:
            formatted_selected_funds = []

        if asset_allocator_option == "Opportunity Scout":
            if fund_type == "Hedge Funds":
                opportunity_scout = OpportunityScout(aws_operations, bucket_name)
                st.title("Opportunity Scout")
                st.markdown("<h3 style='font-size: 20px; color: #6E7C8C;'>Filter for Equity Names Discussed by the Selected {}</h3>".format(fund_type), unsafe_allow_html=True)
                opportunity_scout.run(fund_type, formatted_selected_funds)
            elif fund_type == "Venture Capital Funds":
                st.title("Opportunity Scout")
                st.markdown("<h3 style='font-size: 20px; color: #6E7C8C;'>Filter for Equity Names Discussed by the Selected {}</h3>".format(fund_type), unsafe_allow_html=True)
                vc_opportunity_scout.run(fund_type, formatted_selected_funds)
            else:
                st.write("This feature is not available for the selected fund type.")
        elif asset_allocator_option == "Performance Pulse":
            st.title("Performance Pulse")
            st.markdown("<h3 style='font-size: 20px; color: #6E7C8C;'>Extract Key Insights about your {} Performance</h3>".format(fund_type), unsafe_allow_html=True)
            performance_pulse.run(selected_funds)
        elif asset_allocator_option == "Market Mood Monitor":
            st.title("Market Mood Monitor")
            st.write("Analyze The Sentiment and Perspectives Of The Selected Funds On Various Topics.")
            market_mood_monitor.run(selected_funds)
        elif asset_allocator_option == "Media and Events":
            st.title("Media and Events")
            media_and_events = MediaAndEvents(aws_operations)
            media_and_events.run(selected_funds)

    elif selected_option == "Deep Dive (Single Fund)":
        fund_type = st.sidebar.radio(
            "Select a Fund Type",
            ("Hedge Funds", "Venture Capital Funds", "Private Equity Funds")
        )

        bucket_name = get_bucket_name(fund_type)
        
        if bucket_name:
            fund_info_path = "hedgefund_general_insights.json" if fund_type == "Hedge Funds" else "vc_performance_insights.json"
            fund_names = fetch_fund_names(aws_operations, bucket_name, fund_info_path)
            selected_fund = st.sidebar.selectbox(f"Select a {fund_type}", fund_names)
            
            if fund_type == "Hedge Funds":
                if selected_fund:
                    st.sidebar.write("Please select a fund from the sidebar.")
                specific_funds_section.run(selected_fund)
            elif fund_type == "Venture Capital Funds":
                specific_vc_funds_section.run(selected_fund)
            else:  # Private Equity Funds
                st.write("Logic for PE funds will come soon!")
            
    elif selected_option == "Sources":
        sources_section.run()

def get_bucket_name(fund_type):
    bucket_map = {
        "Hedge Funds": "hedgefunds",
        "Venture Capital Funds": "venturecapitalfunds"
    }
    return bucket_map.get(fund_type)

def select_funds(aws_operations, bucket_name, fund_insights_path):
    fund_names = fetch_fund_names(aws_operations, bucket_name, fund_insights_path)
    fund_names_list = list(fund_names)
    fund_names_list.insert(0, "All")  # Add "All" option at the beginning
    selected_funds = st.sidebar.multiselect("Select Funds", fund_names_list)
    if "All" in selected_funds:
        selected_funds = fund_names_list[1:]  # Select all funds except "All"
    return selected_funds

def format_fund_names(fund_names, fund_type):
    if fund_type == "Hedge Funds":
        return [fund.lower().replace(" ", "") for fund in fund_names]
    else:
        return fund_names

def display_selected_funds(selected_funds):
    if selected_funds:
        st.write("Selected Funds:")
        for fund in selected_funds:
            st.write(fund)
    else:
        st.write("No funds selected.")

if __name__ == '__main__':
    main()