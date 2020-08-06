# split 함수 구현
def split(string, splt_wth):  # 문자열, 나눌 문자(1글자만 가능)
    ans = []
    word = ''  # 임시 문자열
    for s in string+splt_wth:  # 마지막 단어를 위해 splt_wth 더해줌
        if s != splt_wth:  # split문자가 아니면 word에 더해줌
            word = word + s
        else:
            if word != '':  # 빈 문자열이 아니면 정답 리스트에 어펜드해줌
                ans.append(word)
            word = ''  # 다시 초기화
    return ans
