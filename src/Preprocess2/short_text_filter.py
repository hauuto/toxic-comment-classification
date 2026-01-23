import pandas as pd

class ShortTextFilter:
    def __init__(self):
        pass

    def checkLen(self, x: str, threshold: int = 20):
        if not isinstance(x, str):
            return False
        
        if len(x) < threshold: 
            return False
        return True

    def checkSpace(self, x: str, threshold: int = 5):
        if not isinstance(x, str):
            return False
            
        sub = x.split() 
        
        if len(sub) < threshold: 
            return False
        return True

    def filt(self, df: pd.DataFrame, text_col: str, len_threshold: int = 20, space_threshold: int = 5):
        old_cnt = len(df)
        
        texts = df[text_col].values
        
        keep_mask = []
        
        for text in texts:
            is_len_ok = self.checkLen(text, len_threshold)
            is_space_ok = self.checkSpace(text, space_threshold)
            
            if is_len_ok and is_space_ok:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        
        result_df = df[keep_mask].copy()
        
        cur_cnt = len(result_df)
        print(f'Còn lại {cur_cnt}/{old_cnt} sau ShortTextFilter')
        
        return result_df