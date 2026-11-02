"""
TruthLens Automated Verification & Stress Testing Suite
Run this file to verify system accuracy across diverse real-world news categories:
    python3 test_system.py
"""

import time
import app

test_suite = [
    # --- REAL NEWS SCENARIOS ---
    ("Modern Science: James Webb", "NASA James Webb Space Telescope Captures Deepest Image of Early Universe", True),
    ("Space Exploration: Chandrayaan-3", "India's Chandrayaan-3 mission successfully landed near the Moon's south polar region on 23 August 2023. The Vikram lander completed its soft landing.", True),
    ("Renewable Energy: Solar Power", "Headline: Solar Power Becomes One of the World's Fastest-Growing Electricity Sources\n\nSolar energy capacity has expanded significantly in recent years as countries invest in renewable electricity generation.", True),
    ("Medicine: mRNA Vaccines", "Scientists Develop a Vaccine Using Messenger RNA Technology. Messenger RNA technology became widely recognised during the COVID-19 pandemic through vaccines developed by Pfizer-BioNTech and Moderna.", True),
    ("International News: G20 Summit", "India hosts G20 Leaders Summit in New Delhi with adoption of New Delhi Declaration by world leaders", True),
    ("Historical Politics: Reuters 2016", "WASHINGTON (Reuters) - U.S. President Donald Trump removed his chief strategist Steve Bannon from the National Security Council on Wednesday.", True),
    ("Tech: Artificial Intelligence", "DeepMind unveils AlphaFold which predicts 3D structures of proteins from their amino acid sequences", True),
    ("Sports: FIFA World Cup", "Argentina wins 2022 FIFA World Cup final against France in Qatar with Lionel Messi scoring two goals", True),

    # --- FAKE / UNVERIFIED SCENARIOS ---
    ("Pseudoscience: 1-Year Battery", "New Smartphone Battery Technology Allows One-Year Charging. A technology website claims a smartphone battery can operate for one year without recharge. However, it provides no manufacturer announcement, laboratory results or independent testing.", False),
    ("Conspiracy: Green Cheese Moon", "BREAKING: NASA confirms moon is made of green cheese and dairy! Secret space mission reveals government coverup!", False),
    ("Viral Hoax: Pope endorses Trump", "Pope Francis shocks the world and endorses Donald Trump for President of the United States", False),
    ("Health Hoax: Bleach cure", "SHOCKING CURE: Drinking household bleach cures all viral and bacterial infections in 24 hours! Mainstream media won't tell you!", False),
    ("Tech Hoax: 5G causes disease", "Scientific conspiracy exposed: 5G cellular towers transmit biological viruses through radio waves!", False),
    ("Celebrity Hoax: Secret cloning", "SHOCKING PROOF: Government scientists have replaced Hollywood celebrities with synthetic human clones!", False),
    ("Sensational Clickbait: Magic water", "YOU WON'T BELIEVE THIS: Secret miracle water reverses aging instantly! Doctors are furious! 100% proof!", False),
    ("Viral Hoax: Microchips in vaccines", "Leaked documents prove secret tracking microchips were implanted in vaccines to monitor citizens worldwide", False),
]

def run_tests():
    passed = 0
    failed = 0

    print("=" * 80)
    print("                 TRUTHLENS STRESS TEST & EVALUATION SUITE                       ")
    print("=" * 80 + "\n")

    t0 = time.time()
    for name, text, expected_real in test_suite:
        res = app.verify_hybrid(text)
        verdict = res.get("verdict")
        is_real = (verdict == "real")
        is_correct = (is_real == expected_real)

        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct:
            passed += 1
        else:
            failed += 1

        expected_str = "REAL" if expected_real else "FAKE"
        actual_str = "REAL" if is_real else "FAKE"
        print(f"{status} | {name:<35} | Expected: {expected_str:<4} | Got: {actual_str:<4} ({res.get('model')})")

    print("\n" + "=" * 80)
    print(f"Test Suite Summary: {passed}/{len(test_suite)} Passed ({(passed/len(test_suite))*100:.1f}%) in {time.time()-t0:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()

