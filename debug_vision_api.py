"""
Debug Vision API capabilities and find the right object detection endpoint.
"""
import requests
import base64
import os
import json
import sys

def test_all_vision_features():
    """Test all available Vision API features to find object detection."""
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    # Load test image
    image_path = "./assets/WhatsApp Image 2025-10-27 at 22.41.43.jpeg"
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Test different Vision API features
    features_to_test = [
        {"type": "OBJECT_LOCALIZATION", "maxResults": 50},
        {"type": "PRODUCT_SEARCH", "maxResults": 50}, 
        {"type": "LOGO_DETECTION", "maxResults": 50},
        {"type": "LABEL_DETECTION", "maxResults": 50},
        {"type": "TEXT_DETECTION", "maxResults": 50},
        {"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 50},
        {"type": "FACE_DETECTION", "maxResults": 50},
        {"type": "LANDMARK_DETECTION", "maxResults": 50},
        {"type": "IMAGE_PROPERTIES"},
        {"type": "SAFE_SEARCH_DETECTION"},
        {"type": "WEB_DETECTION", "maxResults": 50}
    ]
    
    url = "https://vision.googleapis.com/v1/images:annotate"
    
    results = {}
    
    for i, feature in enumerate(features_to_test):
        feature_type = feature["type"]
        print(f"\n🔍 Testing {i+1:2d}/11: {feature_type}")
        
        payload = {
            "requests": [{
                "image": {"content": image_b64},
                "features": [feature]
            }]
        }
        
        try:
            response = requests.post(f"{url}?key={api_key}", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                response_data = result.get('responses', [{}])[0]
                
                # Count results for each feature
                counts = {}
                total_objects = 0
                
                if 'localizedObjectAnnotations' in response_data:
                    count = len(response_data['localizedObjectAnnotations'])
                    counts['objects'] = count
                    total_objects += count
                    print(f"   ✅ Objects: {count}")
                
                if 'logoAnnotations' in response_data:
                    count = len(response_data['logoAnnotations'])
                    counts['logos'] = count
                    total_objects += count
                    print(f"   ✅ Logos: {count}")
                
                if 'labelAnnotations' in response_data:
                    count = len(response_data['labelAnnotations'])
                    counts['labels'] = count
                    print(f"   ✅ Labels: {count}")
                
                if 'textAnnotations' in response_data:
                    count = len(response_data['textAnnotations'])
                    counts['texts'] = count
                    print(f"   ✅ Texts: {count}")
                
                if 'productSearchResults' in response_data:
                    products = response_data['productSearchResults']
                    count = len(products.get('results', []))
                    counts['products'] = count
                    total_objects += count
                    print(f"   ✅ Products: {count}")
                
                if 'faceAnnotations' in response_data:
                    count = len(response_data['faceAnnotations'])
                    counts['faces'] = count
                    print(f"   ✅ Faces: {count}")
                
                if 'landmarkAnnotations' in response_data:
                    count = len(response_data['landmarkAnnotations'])
                    counts['landmarks'] = count
                    print(f"   ✅ Landmarks: {count}")
                
                if 'webDetection' in response_data:
                    web = response_data['webDetection']
                    web_entities = len(web.get('webEntities', []))
                    print(f"   ✅ Web entities: {web_entities}")
                    counts['web_entities'] = web_entities
                
                if 'error' in response_data:
                    error = response_data['error']
                    print(f"   ❌ Error: {error.get('code', 'Unknown')} - {error.get('message', 'Unknown error')}")
                    counts['error'] = error.get('message', 'Unknown error')
                
                if not counts:
                    print(f"   ⚪ No results (but no error)")
                
                results[feature_type] = {
                    'counts': counts,
                    'total_objects': total_objects,
                    'success': response.status_code == 200,
                    'sample_data': response_data if total_objects > 0 else None
                }
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"   ❌ Failed: {error_msg}")
                results[feature_type] = {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"   ❌ Exception: {error_msg}")
            results[feature_type] = {
                'success': False, 
                'error': error_msg
            }
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 VISION API FEATURE SUMMARY")
    print(f"{'='*80}")
    
    working_features = []
    object_features = []
    
    for feature_type, result in results.items():
        if result['success']:
            total = result.get('total_objects', 0)
            if total > 0:
                object_features.append((feature_type, total))
                print(f"🎯 {feature_type:25s}: {total:3d} objects detected ✅")
            else:
                working_features.append(feature_type)
                print(f"✅ {feature_type:25s}: Working (no objects)")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {feature_type:25s}: {error}")
    
    print(f"\n🔍 OBJECT DETECTION ANALYSIS:")
    
    if object_features:
        print("   ✅ Features that detect objects:")
        for feature, count in sorted(object_features, key=lambda x: x[1], reverse=True):
            print(f"      • {feature}: {count} objects")
        
        # Show best feature details
        best_feature, best_count = object_features[0]
        if best_feature in results and results[best_feature].get('sample_data'):
            print(f"\n📋 Sample from {best_feature}:")
            sample_data = results[best_feature]['sample_data']
            
            if 'localizedObjectAnnotations' in sample_data:
                for i, obj in enumerate(sample_data['localizedObjectAnnotations'][:5]):
                    name = obj.get('name', 'Unknown')
                    score = obj.get('score', 0)
                    print(f"      {i+1}. {name} (confidence: {score:.2f})")
    else:
        print("   ❌ NO OBJECT DETECTION FEATURES WORKING!")
        print("   🔧 Possible issues:")
        print("      • API key doesn't have Vision API enabled")
        print("      • Vision API billing not activated")
        print("      • Need different Google Cloud project")
        print("      • Need service account with proper permissions")
    
    # Save detailed results
    with open("./outputs/vision_api_debug.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: ./outputs/vision_api_debug.json")
    
    return len(object_features) > 0

def check_api_requirements():
    """Check what APIs need to be enabled."""
    print(f"\n{'='*80}")
    print("🔧 GOOGLE CLOUD API REQUIREMENTS")
    print(f"{'='*80}")
    
    print("Für Object Detection brauchen Sie:")
    print("1. 📊 Cloud Vision API - OBJECT_LOCALIZATION")
    print("   • Google Cloud Console → APIs & Services → Library")
    print("   • Suchen: 'Cloud Vision API'")
    print("   • ENABLE klicken")
    print()
    print("2. 💰 Billing Account aktivieren")
    print("   • Google Cloud Console → Billing")
    print("   • Link project to billing account")
    print()
    print("3. 🔑 API Key mit richtigen Permissions")
    print("   • Google Cloud Console → APIs & Services → Credentials")
    print("   • API Key → Edit → API restrictions")
    print("   • 'Cloud Vision API' auswählen")
    print()
    print("4. 🏪 Optional: Product Search API für Retail")
    print("   • Für bessere Produkt-Erkennung")
    print("   • Separate API activation nötig")

if __name__ == "__main__":
    print("🔍 Vision API Debugging Tool")
    print("Testing all available Vision API features...")
    
    os.makedirs("./outputs", exist_ok=True)
    
    success = test_all_vision_features()
    check_api_requirements()
    
    print(f"\n{'='*80}")
    if success:
        print("✅ OBJECT DETECTION GEFUNDEN! Schauen Sie die Details oben.")
    else:
        print("❌ OBJECT DETECTION NICHT VERFÜGBAR!")
        print("   Folgen Sie den API Requirements oben.")
    print(f"{'='*80}")