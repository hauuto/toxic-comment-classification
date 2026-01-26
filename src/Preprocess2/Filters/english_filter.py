import langdetect
import pandas as pd
from langdetect import LangDetectException

class EnglishFilter:
    def __init__(self):
        langdetect.DetectorFactory.seed = 0

    def getViChance(self, text: str):
        try:
            text = str(text)
            if not text.strip():
                return 0.0
                
            res = langdetect.detect_langs(text)
            for i in res:
                if i.lang == 'vi':
                    return i.prob
            return 0.0
        except LangDetectException:
            return 0.0

    def isVi(self, text: str):
        try:
            text = str(text)
            if not text.strip():
                return False
            return langdetect.detect(text) == 'vi'
        except LangDetectException:
            return False
    
    def filt(self, df: pd.DataFrame, text_col: str):
        old_cnt = len(df)
        
        texts = df[text_col].values
        
        keep_mask = []   
        vi_chances = []  
        
        for text in texts:
            is_vietnamese = self.isVi(text)
            
            if is_vietnamese:
                keep_mask.append(True)
                vi_chances.append(self.getViChance(text))
            else:
                keep_mask.append(False)
        
        result_df = df[keep_mask].copy()
        result_df['vi_chance'] = vi_chances
        result_df['is_vi'] = True 
        
        cur_cnt = len(result_df)
        print(f'Còn lại {cur_cnt}/{old_cnt} sau EnglishFilter')
        
        return result_df
