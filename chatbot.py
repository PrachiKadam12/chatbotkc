
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")
print(data)

print("welcome to kamal classes helpline --> I am chitty (press q for quit )")

while True:
    qts = input("enter question --> ").strip().lower()
    if qts == "q":
        break
    else:
        texts = [qts] + data["question"].str.lower().tolist()
        print(texts)   # humara qts + already given qts ka corpus
        input()

        cv = CountVectorizer()
        vector = cv.fit_transform(texts)   # pura corpus vectorize ho gaya
        print(vector)
        input()

        cs = cosine_similarity(vector)   # so that we can find similarity
        score = cs[0][1:]
        data["score"] = score * 100
        print(data)
        input()

        result = data.sort_values(by="score", ascending=False)   # result sort from highest to lowest
        print(result)
        input()

        result = result[result.score > 10]   # woh sab result rahenge jisko score > 10
        if len(result) == 0:
            print("chitty --> sorry i dont know please contact - 7498405040 ")
        else:
            # highest wala
            ans = result.head(1)["answer"].values[0]
            print("chitty --> ", ans)

        # randomize answer
        # a1 = result.sample(1)
        # ans = a1.head(1)["answer"].values[0]
        # print("chitty --> ", ans)