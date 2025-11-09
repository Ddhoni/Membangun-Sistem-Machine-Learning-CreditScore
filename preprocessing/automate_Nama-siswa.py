#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
automate_Nama-siswa.py
Pipeline preprocessing otomatis berdasarkan eksperimen di notebook.
Menyiapkan data train/test siap latih.

Contoh pakai:
python automate_Nama-siswa.py \
  --train "../data/train.csv" \
  --test  "../data/test.csv" \
  --outdir "../data_preprocessed"
"""

import os
import re
import argparse
import pandas as pd
import numpy as np


# ============== Util umum ==============

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def text_cleaning(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        return val.strip('_ ,"')
    return val


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan simbol aneh di object-col & konversi tipe sesuai eksperimen."""
    data = df.copy()

    # Bersih simbol pada kolom object
    obj_cols = data.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        data[obj_cols] = data[obj_cols].apply(lambda col: col.map(text_cleaning))
        data[obj_cols] = data[obj_cols].replace(
            ['', 'nan', '!@9#%8', '#F%$D@*&8'],
            np.nan
        )

    # ID dari hex → int
    if 'ID' in data.columns:
        data['ID'] = data['ID'].apply(lambda x: int(x, 16) if isinstance(x, str) else (x if pd.isna(x) else np.nan))

    # Customer_ID dari offset hex → int
    if 'Customer_ID' in data.columns:
        data['Customer_ID'] = data['Customer_ID'].apply(
            lambda x: int(x[4:], 16) if isinstance(x, str) and len(x) > 4 else (x if pd.isna(x) else np.nan)
        )

    # Month "January" → 1..12
    if 'Month' in data.columns:
        data['Month'] = pd.to_datetime(data['Month'], format='%B', errors='coerce').dt.month

    # Age (nullable int)
    if 'Age' in data.columns:
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce').astype('Int64')

    # SSN → float tanpa '-'
    if 'SSN' in data.columns:
        def _ssn_cast(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (str, int, float)):
                return float(str(x).replace('-', ''))
            return np.nan
        data['SSN'] = data['SSN'].apply(_ssn_cast)

    # Kolom numerik lainnya → numeric
    num_cols = [
        'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance',
        'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Delay_from_due_date', 'Num_Credit_Inquiries',
        'Total_EMI_per_month'
    ]
    for col in num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data


def preprocess_credit_and_loan(df: pd.DataFrame) -> pd.DataFrame:
    """Parsing Credit_History_Age ke bulan & normalisasi Type_of_Loan."""
    data = df.copy()

    def month_converter(x):
        if pd.isna(x):
            return np.nan
        s = str(x).lower()
        try:
            s = s.replace('years','year').replace('months','month')
            parts = s.split()
            nums = [int(tok) for tok in parts if tok.isdigit()]
            if len(nums) >= 2:
                years, months = nums[0], nums[1]
                return float(years * 12 + months)
            if len(nums) == 1:
                return float(nums[0])
        except Exception:
            pass
        return np.nan

    if 'Credit_History_Age' in data.columns:
        data['Credit_History_Age'] = data['Credit_History_Age'].apply(month_converter).astype(float)

    if 'Type_of_Loan' in data.columns:
        data['Type_of_Loan'] = data['Type_of_Loan'].apply(
            lambda x: x.lower().replace('and ', '').replace(', ', ',').strip()
            if pd.notna(x) else x
        )

    return data


def fill_object_nan_by_group_mode(
    df: pd.DataFrame,
    group_col: str,
    columns: list,
) -> pd.DataFrame:
    """Isi NaN object-cols per group dengan modus grup (sesuai eksperimen)."""
    work = df.copy()
    for col in columns:
        if col not in work.columns:
            continue
        mask_none = work[col].isin([None])
        if mask_none.any():
            work.loc[mask_none, col] = np.nan
        mode_per_group = (
            work.groupby(group_col)[col]
                .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        work[col] = work[col].fillna(mode_per_group)
    return work


def Numeric_Wrong_Values_Reassign_Group_Min_Max(data, groupby, column):
    """
    Bersihkan nilai numerik di 'column' per 'groupby':
      - Range min–max dari mode per-grup (fallback quantile 5–95 → min/max global → 0)
      - Out-of-range → NaN
      - Imput: mode per-grup → median per-grup → median global → 0
    """
    work = data.copy()
    work[column] = pd.to_numeric(work[column], errors='coerce')

    cur = (
        work.loc[work[column].notna()]
            .groupby(groupby)[column]
            .apply(list)
    )
    if len(cur) > 0:
        mode_vals = []
        for vals in cur:
            m = pd.Series(vals).mode()
            if not m.empty:
                mode_vals.append(m.iloc[0])
        if len(mode_vals) > 0:
            mini, maxi = float(np.min(mode_vals)), float(np.max(mode_vals))
        else:
            mini = maxi = np.nan
    else:
        mini = maxi = np.nan

    s_non_na = work[column].dropna()
    if (not np.isfinite(mini)) or (not np.isfinite(maxi)):
        if not s_non_na.empty:
            mini = float(s_non_na.quantile(0.05))
            maxi = float(s_non_na.quantile(0.95))
        else:
            mini = maxi = 0.0
    if mini > maxi and not s_non_na.empty:
        mini, maxi = float(s_non_na.min()), float(s_non_na.max())

    marked = work[column].where(~((work[column] < mini) | (work[column] > maxi)), np.nan)

    mode_by_group = work.groupby(groupby)[column].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    result = marked.fillna(mode_by_group)

    if result.isna().any():
        median_by_group = work.groupby(groupby)[column].transform(lambda x: x.median())
        result = result.fillna(median_by_group)

    if result.isna().any():
        global_med = work[column].median()
        if pd.isna(global_med):
            global_med = 0.0
        result = result.fillna(global_med)

    result = result.fillna(0.0)
    work[column] = result
    return work


def clean_numeric_columns(df: pd.DataFrame, group_col='Customer_ID') -> pd.DataFrame:
    """Bersihkan kolom numerik + interpolasi Credit_History_Age per grup."""
    work = df.copy()
    numeric_cols = [
        'Age','SSN','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
        'Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date',
        'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
        'Outstanding_Debt','Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance'
    ]
    for col in numeric_cols:
        if col in work.columns:
            work = Numeric_Wrong_Values_Reassign_Group_Min_Max(work, group_col, col)

    if 'Credit_History_Age' in work.columns and group_col in work.columns:
        work['Credit_History_Age'] = (
            pd.to_numeric(work['Credit_History_Age'], errors='coerce')
              .groupby(work[group_col])
              .transform(lambda s: s.interpolate().bfill().ffill())
        )
    return work


def fix_strange_values(df: pd.DataFrame, group_col: str = 'Customer_ID') -> pd.DataFrame:
    """Perbaiki nilai aneh sesuai aturan eksperimen."""
    work = df.copy()

    if 'Num_Bank_Accounts' in work.columns:
        work.loc[work['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = 0

    if 'Delay_from_due_date' in work.columns:
        work.loc[work['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = np.nan
        work = Numeric_Wrong_Values_Reassign_Group_Min_Max(work, group_col, 'Delay_from_due_date')

    if 'Num_of_Delayed_Payment' in work.columns:
        work.loc[work['Num_of_Delayed_Payment'] < 0, 'Num_of_Delayed_Payment'] = np.nan
        work = Numeric_Wrong_Values_Reassign_Group_Min_Max(work, group_col, 'Num_of_Delayed_Payment')

    if 'Monthly_Balance' in work.columns:
        work.loc[work['Monthly_Balance'] < 0, 'Monthly_Balance'] = np.nan
        work = Numeric_Wrong_Values_Reassign_Group_Min_Max(work, group_col, 'Monthly_Balance')

    if 'Amount_invested_monthly' in work.columns:
        work.loc[work['Amount_invested_monthly'] >= 10000, 'Amount_invested_monthly'] = np.nan
        if group_col in work.columns:
            work['Amount_invested_monthly'] = work.groupby(group_col)['Amount_invested_monthly'].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
    return work


def encode_categorical_columns(df: pd.DataFrame, columns: list = None, suffix: str = "_Num") -> pd.DataFrame:
    """Kategori → category codes (membuat kolom baru dengan suffix)."""
    work = df.copy()
    if columns is None:
        columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    for col in columns:
        if col in work.columns:
            new_col = f"{col}{suffix}"
            work[new_col] = (
                work[col].astype('category').cat.codes.replace(-1, np.nan)
            )
    return work


def full_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline lengkap sesuai notebook."""
    data = clean_and_transform(df)
    data = preprocess_credit_and_loan(data)

    # Siapkan 'Type_of_Loan' NaN → "No Data" (sesuai eksperimen)
    if 'Type_of_Loan' in data.columns:
        data['Type_of_Loan'] = data['Type_of_Loan'].fillna('No Data')

    # Isi NaN object by-group mode
    if 'Customer_ID' in data.columns:
        data = fill_object_nan_by_group_mode(
            data,
            group_col='Customer_ID',
            columns=['Name','Occupation','Credit_Mix','Payment_Behaviour']
        )

    # Bersihkan numerik + interpolasi CHA
    data = clean_numeric_columns(data, group_col='Customer_ID')

    # Perbaiki nilai "strange"
    data = fix_strange_values(data, group_col='Customer_ID')

    # Encode kategori → codes
    data = encode_categorical_columns(data)

    return data


def split_and_save(processed: pd.DataFrame, outdir: str, target_col: str = 'Credit_Score'):
    """Pisahkan kembali ke train/test berdasarkan ketersediaan target, lalu simpan."""
    ensure_dir(outdir)

    if target_col in processed.columns:
        train_out = processed[processed[target_col].notna()].copy()
        test_out  = processed[processed[target_col].isna()].drop(columns=target_col).copy()
    else:
        # Bila target tidak ada, simpan sebagai satu berkas processed.csv
        processed.to_csv(os.path.join(outdir, "processed.csv"), index=False)
        return

    train_out.to_csv(os.path.join(outdir, "preprocessed_train.csv"), index=False)
    test_out.to_csv(os.path.join(outdir, "preprocessed_test.csv"), index=False)


# ============== CLI ==============

def parse_args():
    p = argparse.ArgumentParser(description="Automated preprocessing pipeline (siap latih).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--combined", type=str, help="Path ke file CSV gabungan (train+test).")
    g.add_argument("--train", type=str, help="Path ke train.csv.")
    p.add_argument("--test", type=str, help="Path ke test.csv (opsional bila pakai --train).")
    p.add_argument("--outdir", type=str, required=True, help="Folder output hasil preprocessing.")
    p.add_argument("--split-col", type=str, default="Credit_Score",
                   help="Nama kolom target untuk memisahkan train/test dari combined.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.combined:
        df = pd.read_csv(args.combined)
        processed = full_preprocess(df)
        split_and_save(processed, args.outdir, target_col=args.split_col)
        print(f"✅ Selesai! Hasil disimpan ke: {args.outdir}")
        return

    # mode train+test terpisah
    train_df = pd.read_csv(args.train)
    if args.test:
        test_df  = pd.read_csv(args.test)
        # gabung → proses → pecah lagi mengikuti keberadaan target
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df

    processed = full_preprocess(df)
    split_and_save(processed, args.outdir, target_col="Credit_Score")
    print(f"✅ Selesai! Hasil disimpan ke: {args.outdir}")


if __name__ == "__main__":
    main()