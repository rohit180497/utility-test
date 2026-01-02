import os
import json
import ssl
import httpx
from time import perf_counter
from dotenv import load_dotenv
from openai import AzureOpenAI  
from utility import PIIUtility  
import asyncio  

# Helper function to call Azure OpenAI
# This function sends anonymized text to Azure OpenAI and retrieves the response.
def call_azure_openai(endpoint, api_key, deployment, api_version, input_text):
    """
    Call Azure OpenAI ChatCompletion with SSL verification disabled.
    """
    try:
        # Create an insecure SSL context for testing purposes
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Pass the insecure SSL context to the HTTP client for testing purposes
        http_client = httpx.Client(verify=ssl_context)

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,  
            http_client=http_client  # Pass the insecure client
        )

        response = client.chat.completions.create(
            model=deployment,  # Updated to use deployment name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ],
            max_completion_tokens=10000,  # Added additional parameters
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Azure OpenAI call failed: {e}")
        return None


if __name__ == "__main__":
    async def main():
        # Load environment variables from .env file
        try:
            load_dotenv()
        except Exception:
            pass

        # Retrieve Azure OpenAI credentials from environment variables
        AZ_MODEL = os.getenv("AZURE_OPENAI_MODEL")
        AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZ_EP = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZ_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  # Added API version

        # Initialize the PIIUtility for anonymization and de-anonymization
        pii_utility = PIIUtility(model_path="./en_core_web_lg-3.8.0", persist=False, allowed_text=["GB"], basic_pii=True, medical_pii=True)

        # Sample text containing PII
        sample_text = (
            "Describe this situation: In Premera Blue Cross, Patient John Maria with was admitted on January 15th. Weather is good today with 50 degrees, yesterday on 24/12/2025 it was 40 degrees. "
            "Subscriber ID: 98776567889. Contact: john.maria@gmail.com, phone +1 (555) 123-4567. SSN 123-45-6789 and member id 90867987 Michael Hussy., mobile no. 857-424-4107")
        

        print("Original Text:\n", sample_text)

        # Step 1: Anonymize the text (measure execution time)
        t_start = perf_counter()
        anonymized_text, mapping_id = await pii_utility.mask(sample_text, store_mapping=True)
        elapsed = perf_counter() - t_start
        print(f"\nAnonymized Text (in {elapsed*1000:.2f} ms):\n", anonymized_text)
        print("Mapping ID:", mapping_id)

        # Step 2: Send anonymized text to Azure OpenAI (or simulate response if credentials are missing)
        if AZ_KEY and AZ_EP and AZ_MODEL and AZ_API_VERSION:
            print("\nCalling Azure OpenAI...")
            llm_response = call_azure_openai(AZ_EP, AZ_KEY, AZ_MODEL, AZ_API_VERSION, anonymized_text)
            if not llm_response:
                llm_response = anonymized_text
        else:
            print("\nAzure credentials not configured; using simulated LLM response.")
            llm_response = anonymized_text + "\nNote: follow up with __PII_EMAIL_1__"

        # print("\nLLM Response:\n", llm_response)

        # Step 3: De-anonymize the LLM response
        de_anonymized_text = await pii_utility.unmask(llm_response, mapping_id=mapping_id)
        # print("\nDe-anonymized Text:\n", de_anonymized_text)

        # Step 4: Retrieve the mapping and convert it to a structured PII dictionary
        pii_mapping = await pii_utility.get_mapping(mapping_id)
        pii_dict = pii_utility.mapping_to_pii_dict(pii_mapping)

        # Step 5: Build the final JSON output
        final_output = {
            "text": de_anonymized_text,
            "PII": pii_dict,
            "mapping_id": mapping_id,
        }

        print("\nFinal JSON Output:\n", json.dumps(final_output, ensure_ascii=False, indent=2))

    # Run the async main function
    asyncio.run(main())
