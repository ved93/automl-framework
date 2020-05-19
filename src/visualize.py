
import pandas as pd
import pandas_profiling 

# TRAINING_DATA = os.environ.get("TRAINING_DATA")

    
def create_summary_report(df):
    #Data profiling/EDA 
    profile = pandas_profiling.ProfileReport(df, title='Data Audit Report \nAuthor: Ved')
    profile.to_file(output_file="../reports/data_audit_report.html")
    



if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")

    create_summary_report(df)
