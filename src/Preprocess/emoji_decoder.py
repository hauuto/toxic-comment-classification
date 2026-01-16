import emoji

def emoji_decode(x: str):
    '''
        Input: Chuỗi + Emoji
        Output: Chuỗi + Decoded emote
    '''    
    res = ''
    for char in x:
        if emoji.is_emoji(char):
            char = emoji.demojize(char)
        res += char
    return res

emoji_decode("đã nghe one of the girls và thấy bài này hay hơn chán🥰")