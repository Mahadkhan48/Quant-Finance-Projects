

from pathlib import Path


import pandas as pd


def main():
   

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\Faccini_Climate_Risk_Monthly.xlsx"
    )
   

    output_file = input_file.parent / "Faccini_Climate_Risk_Monthly_With_Indices.xlsx"
 

    df = pd.read_excel(input_file)
    

    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    

    df["Physical Risk"] = (df["Global warming"] + df["Natural disasters"]) / 2
   

    df["Transition Risk"] = (df["US climate policy"] + df["International summits"]) / 2
    

    df["Aggregate Climate Risk"] = (
        df["US climate policy"]
        + df["International summits"]
        + df["Global warming"]
        + df["Natural disasters"]
    ) / 4
    

    df.to_excel(output_file, index=False)
    
    print(f"Output file saved successfully at:\n{output_file}")
   

if __name__ == "__main__":
   

    main()
    