#!/usr/bin/env python3
"""
Quick Test for new Google API Key
"""
import os
import json
import base64
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv()

def test_google_api():
    """Test both Vision and Gemini APIs with new key"""
    
    api_key = os.getenv('GOOGLE_API_KEY')
    print(f"🔑 Testing API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # 1. Test Vision API
    print("\n🔍 Testing Vision API...")
    
    # Create a simple test image (1x1 pixel base64)
    test_image_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    
    vision_request = {
        "requests": [{
            "image": {"content": test_image_b64},
            "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 5}]
        }]
    }
    
    try:
        vision_response = requests.post(vision_url, 
                                     json=vision_request, 
                                     timeout=10)
        
        if vision_response.status_code == 200:
            print("✅ Vision API: OK")
        else:
            print(f"❌ Vision API Error: {vision_response.status_code}")
            print(f"Response: {vision_response.text}")
            
    except Exception as e:
        print(f"❌ Vision API Exception: {e}")
    
    # 2. Test Gemini API
    print("\n🤖 Testing Gemini API...")
    
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    
    gemini_request = {
        "contents": [{
            "parts": [{
                "text": "Hello! Just testing if the API works. Please respond with 'API Working'"
            }]
        }]
    }
    
    try:
        gemini_response = requests.post(gemini_url, 
                                      json=gemini_request, 
                                      timeout=10)
        
        if gemini_response.status_code == 200:
            response_data = gemini_response.json()
            if 'candidates' in response_data:
                print("✅ Gemini API: OK")
                text_response = response_data['candidates'][0]['content']['parts'][0]['text']
                print(f"   Response: {text_response.strip()}")
            else:
                print("❌ Gemini API: Unexpected response format")
        else:
            print(f"❌ Gemini API Error: {gemini_response.status_code}")
            print(f"Response: {gemini_response.text}")
            
    except Exception as e:
        print(f"❌ Gemini API Exception: {e}")
    
    print("\n🎯 API Test Complete!")

if __name__ == "__main__":
    test_google_api()