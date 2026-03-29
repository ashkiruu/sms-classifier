"""Evaluate saved models on the 15-record confidence set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_service import predict_with_ensemble, predict_with_nn
from utils import get_report_path, setup_logger

logger = setup_logger(__name__)

RECORDS = [
    {"date": "23/09/2024 16:17", "sender": "BDO Deals", "text": "Earn Peso Points effortlessly! Simply Scan to Pay with BDO Pay in stores and get 50 Peso Points for every 1,000 spent from August 15 to October 31, 2024. Download BDO Pay! T&Cs apply. DTI200125", "label": "ads"},
    {"date": "22/09/2024 18:01", "sender": "GCash", "text": "Did you request to SEND MONEY to CH**O NI***E E.'s GCash number, 09706864475 with amount of PHP 190.00? If not, DON'T ENTER YOUR OTP ON ANY SITE OR SEND IT TO ANYONE because IT'S A SCAM! If you requested, your OTP is 652025.", "label": "otp"},
    {"date": "22/09/2024 12:48", "sender": "TNT", "text": "DOBLENG SAYA, DOBLENG PANALO! HANGGANG TODAY NA LANG: sagot na namin ang 2X DATA mo sa TIKTOK SAYA 50 (3 GB + FREE 3 GB)! 2X DATA ka na rin sa: SURFSAYA 30/49/99 ALL DATA 50/99 ALL DATA+ 75/149 TIKTOK SAYA 99/149 GIGA VIDEO/STORIES/GAMES 60/120 DOBLE GIGA VIDEO+/STORIES+ 75/149 TRIPLE DATA VIDEO+/STORIES+ 75/149 UTP+ 30 Kaya load na via https://smrt.ph/GetSmartApp o i-dial ang *123#", "label": "notifs"},
    {"date": "05/09/2024 2:31", "sender": "TingogTayo", "text": "ISIP Beneficiary Message Part 2 of 2 Plus many more prizes to be given away :) Please be sure to bring the following: 1. Wear you ISIP bracelet for entry 2. Screenshot of your registered ISIP IDs 3. Valid ID (Original with 2 Xerox Copies, eg. validated school ID and any Valid Government ID) 4. Original & 1 Xerox Copy of Barangay Certificate (indicating the following ONLY: that you are an indigent or belong to the indigent group, for the purpose of DSWD financial assistance) See you there!", "label": "gov"},
    {"date": "03/09/2024 18:47", "sender": "BagongPinas", "text": "Ang iyong One Time Pin ay: 191023", "label": "gov"},
    {"date": "22/08/2024 13:00", "sender": "6.39621E+11", "text": "Manatili lamang sa bahay,tuturuan kita,araw-araw 1000+,Telegram:@apk552", "label": "spam"},
    {"date": "22/08/2024 10:56", "sender": "BDO", "text": "APP-GRADE to the new BDO Online app to continue to view your account balances and make transactions. Download the BDO Online app now!", "label": "notifs"},
    {"date": "21/08/2024 14:55", "sender": "CIMB_Bank", "text": "CIMB MaxSave is now available for a shorter 3-month term. Start saving now for a minimum deposit of ₱5,000! Visit CIMB website to learn more.", "label": "ads"},
    {"date": "21/08/2024 0:44", "sender": "3404", "text": "Use 868178 for two-factor authentication on Facebook.", "label": "otp"},
    {"date": "24/06/2024 1:35", "sender": "6.39703E+11", "text": "Spin the Daily Lucky Wheel atGojplaywild.de and win a Samsung Galaxy S23+ 5G worth P42,999! New members grab a 130% bonus up to P1000. Don't miss out!", "label": "spam"},
    {"date": "20/06/2024 17:28", "sender": "6.3965E+11", "text": "Join the Slot Tournament now! Get 3000p FREE on sign-up + a chance to win a VIVO V30e 256GB! Don't miss out on the excitement! sbetphlucks.eu", "label": "spam"},
    {"date": "17/06/2024 14:35", "sender": "BDO Deals", "text": "Get up to 1,000 bonus Peso Points with BDO Pay! Simply Scan to Pay to earn 2 bonus Peso Points per transaction. Promo runs from June 17 to July 31, 2024. Visit the BDO website, click Deals and search Bonus Points to learn more. T&Cs apply. DTI195469", "label": "ads"},
    {"date": "14/06/2024 21:14", "sender": "TNT", "text": "100% Cashback handog ng TNT at Maya! I-download at i-upgrade ang Maya app at bumili ng P100 load for yourself sa Maya. I-claim mo na bago pa mawala: https://official.maya.ph/3xMF/SmartSignUp T&Cs apply. Users who qualify for the promo will receive their reward within 3 business days. DTI192508", "label": "notifs"},
    {"date": "21/12/2023 12:39", "sender": "NTC", "text": "1/3 This is a public service advisory from the National Telecommunications Commission, Telecommunications Connectivity, Inc. and SMART.", "label": "gov"},
    {"date": "27/10/2023 15:48", "sender": "GOMO", "text": "GOMO Anniv ain't over yet! Get a FREE UPsize from PICKUP COFFEE tomorrow, October 28! Just show your unique referral code found in the GOMO PH app, under Account > Refer a Friend. Valid in selected PICKUP COFFEE branches nationwide. Visit https://bit.ly/GOMOFAQTopic for more info. No advisories? Text OFF to 2686 for free.", "label": "ads"},
]


def main() -> None:
    rows = []
    for record in RECORDS:
        ensemble = predict_with_ensemble(record["text"], sender=record["sender"])
        nn = predict_with_nn(record["text"], sender=record["sender"])
        rows.append(
            {
                **record,
                "ensemble_prediction": ensemble["prediction"],
                "ensemble_confidence": ensemble["confidence"],
                "nn_prediction": nn.get("prediction"),
                "nn_confidence": nn.get("confidence"),
                "nn_available": nn.get("available", True),
            }
        )

    df = pd.DataFrame(rows)
    true_labels = df["label"].tolist()

    ensemble_acc = accuracy_score(true_labels, df["ensemble_prediction"].tolist())
    get_report_path("ensemble_confidence_set_predictions.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    get_report_path("ensemble_confidence_set_classification_report.txt").write_text(
        classification_report(true_labels, df["ensemble_prediction"].tolist(), digits=4),
        encoding="utf-8",
    )

    if df["nn_available"].all():
        nn_acc = accuracy_score(true_labels, df["nn_prediction"].tolist())
        get_report_path("confidence_set_predictions.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        get_report_path("confidence_set_classification_report.txt").write_text(
            classification_report(true_labels, df["nn_prediction"].tolist(), digits=4),
            encoding="utf-8",
        )
    else:
        nn_acc = None

    summary = {"ensemble_accuracy": ensemble_acc, "nn_accuracy": nn_acc}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
