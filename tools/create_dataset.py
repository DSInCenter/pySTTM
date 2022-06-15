"""
    This is the function of the scraper that generates a dataset from a list of hashtags.
"""

from scraper import TwitterScraper
from hazm import word_tokenize, Normalizer, Lemmatizer
import numpy as np
import pandas as pd
import argparse

normalizer = Normalizer().normalize
lemmatizer = Lemmatizer().lemmatize

# Retrieved from https://github.com/kharazi/persian-stopwords
stopwords = set(open('../stop_words/stop_words.txt', encoding='utf8').read().splitlines())
# Retrieved from https://github.com/amirshnll/Persian-Swear-Words
swearing_words = set(open('../stop_words/swearing_words.txt', encoding='utf8').read().splitlines())

bad_hashtags = set(['تا_آخوند_کفن_نشود_این_وطن_وطن_نشود',
'ایران_را_پس_میگیریم',
'جمهوری_اسلامی_نابود_باید_گردد',
'مرگ_بر_خامنه\\u200cای_جنایتکار',
'مرگ_بر_کلیت_و_تمامیت_جمهوری_اسلامی',
'جاویدشاه',
'نه_به_جمهورى_اسلامى',
'ریدم_تو_اسلام',
'براندازم',
'قيام_تا_سرنگونی',
'مريم_رجوی'])

swearing_words.update(bad_hashtags)

class const:
    farsi = ('ب', 'س', 'ش', 'ل', 'ت', 'ن', 'م', 'گ', 'ظ', 'ط', 'ز',
             'ر', 'ژ', 'ذ', 'د', 'پ', 'چ', 'ج', 'ح', 'ع', 
             'خ', 'غ', 'ف', 'ق', 'ث', 'ص', 'ض','\u0020',
             '\u200C', '\u060c','؟', '!', '?', '.', ':','\n', '_')

    alef = ('ا', 'آ', 'ء', 'أ', 'إ')
    vav = ('و', 'ؤ')
    heh = ('ه', 'ة', 'ە')
    yah = ('ی', 'ي', 'ئ', 'ى')
    kaf = ('ک', 'ك')

hashtags = {
    "economics": [
        "بورس",
        "نفت",
        "دلار",
        "بازارکار",
        "اقتصادی",
        "اخبار_اقتصادی",
        "اقتصاد_ایران",
        "بازار_آزاد",
        "بانک_مرکزی",
        "ارز",
        "مالیات",
        "تورم",
        "نرخ",
        "تحریم",
        "طلا",
        "ارز۴۲۰۰",
        "گرانی",
        "بانک",
        "سهام_عدالت",
        "خودرو",
        "فارکس",
        "بنزین",
        "بازار",
        "نرخ_ارز",
        "یورو",
        "قیمت_نفت",
        "بودجه",
        "قیمت",
        "بازار_کار",
        "اقتصاد",
        "سکه",
        "فرابورس",
        "سهام",
        "بیمه",
    ],
    "health": [
        "کرونا",
        "وزارت_بهداشت",
        "نه_به_واکسن_اجباری",
        "واکسن",
        "واکسن_بزنیم",
        "كرونا",
        "اومیکرون",
        "پزشکی",
        "واکسن_اجباری",
        "واکسن_کرونا",
        "پزشک",
        "امیکرون",
        "واکسیناسیون",
        "ماسک",
        "آمار_کرونا",
        "واکسن_میزنم",
        "وزات_بهداشت",
        "بهداشت",
        "کووید۱۹",
        "COVID19",
        "وزیر_بهداشت",
        "HIV",
        "اميكرون",
        "نه_به_واکسن",
        "بهترین_واکسن_در_دسترس_ترین_واکسن",
        "أوميكرون",
        "واکسن_حق_مردم",
        "واكسن",
        "برکت",
    ],
    "sport": [
        "استقلال",
        "پرسپولیس",
        "فوتبال",
        "پرسپوليس",
        "ورزش",
        "HalaMadrid",
        "رئال_مادرید",
        "ورزش_سیاسی_نیست",
        "لیگ_برتر",
        "تیم_حکومتی",
        "تاج",
        "آرسنال",
        "پیروزی",
        "فرهاد_مجیدی",
        "والیبال",
        "المپیک",
        "حامد_لک",
        "فوتبال_پاک",
        "دربی",
        "فیفا",
        "لیورپول",
        "پنالتی",
        "فنرباغچه",
        "تراکتور",
        "لیگ",
        "فدراسیون_آبی",
        "ورزش_سیاسی",
        "چلسی",
        "RealPSG",
        "جام_جهانی",
        "مهدی_طارمی",
        "تیم",
        "تنیس",
        "باشگاه",
    ],
    "art": [
        "شعر",
        "کتاب",
        "سینما",
        "تئاتر",
        "فیلم",
        "سریال",
        "كتاب",
        "موسیقی",
        "پیشنهاد_فیلم",
        "آهنگ",
        "حافظ",
        "سعدی",
        "معرفی_کتاب",
        "کارگردان",
        "خواننده",
        "جشنواره_فیلم_فجر",
        "film",
        "cinema",
        "actor",
        "drama",
        "moviestar",
        "Movietime",
    ],
    "tech": [
        "اینترنت",
        "اپل",
        "سامسونگ",
        "بازی",
        "گیم",
        "گوگل",
        "بیت_کوین",
        "کریپتو",
        "اتریوم",
        "ارزدیجیتال",
        "BTC",
        "همراه_اول",
        "Bitcoin",
        "ارز_دیجیتال",
        "بيتكوين",
        "سئو",
        "بیتکوین",
        "ایرانسل",
        "btc",
        "کاردانو",
        "دیجیکالا",
        "هوشمند",
        "استارلینک",
    ],
    "transport": [
        "ترافیک",
        "اسنپ",
        "تپسی",
        "تاکسی",
        "هواپیما",
        "مترو",
        "اتوبوس",
        "طرح_ترافیک",
        "قطار",
        "فرودگاه",
        "سفر_استانی",
        "فرودگاه_مهرآباد",
        "جاده_چالوس",
    ],
    "education": [
        "معلم",
        "آموزش",
        "دانشگاه",
        "کنکور",
        "دانشگاه_آزاد",
        "مدرسه",
        "دانش_آموز",
        "کنکور_سراسری",
        "سازمان_سنجش",
        "دانشگاه_تهران",
        "آموزش_و_پرورش",
        "دانشجو",
        "معلمان",
        "روز_معلم",
        "فرهنگیان",
        "مدارس",
        "دانشگاه_فرهنگیان",
    ],
    "religion": [
        "یا_سید_الساجدین",
        "امام_سجاد",
        "اللهم_عجل_لوليك_الفرج",
        "امام_حسین",
        "خدا",
        "امام",
        "رمضان",
        "قرآن",
        "مسلمان",
        "اسلام",
        "عاشورا",
        "شیعه",
        "حج",
        "MuhammadForAll",
        "زين_العابدين",
        "امام_رضا",
    ],
    "lifestyle": [
        "شیک",
        "زیبایی",
        "تقویم_آشپزی",
        "پوست",
        "آشپزی",
        "غذا",
        "قهوه",
        "رستوران",
    ],
    "social": [
        "روز_جهانی_زن",
        "زن",
        "زنان",
        "روز_زن",
        "خانواده",
        "کشف_حجاب",
        "هشتم_مارس",
        "باحجاب_باوقار",
        "خودکشی",
        "ازدواج",
        "طلاق",
        "فقر",
        "مردان",
        "کودک_همسری",
        "زندانی_سیاسی",
        "حقوق_زنان",
        "حجاب",
    ],
    "ecology": [
        "باران",
        "هوا",
        "آب",
        "زلزله",
        "کم_آبی",
        "آلودگی_هوا",
        "آلودگی",
        "ریزگرد",
        "هوای_تهران",
        "کولاک",
        "گردوخاک",
        "گردوغبار",
        "بارش",
        "سیلاب",
        "بارندگی",
        "آلودگی_هوای_تهران",
        "مدیریت_بحران",
        "برف",
        "سیل",
        "آتش",
        "آتش_سوزی",
        "خشکسالی",
        "محیط_زیست",
        "خاک",
        "هواشناسى",
        "هواشناسی_توییتر",
    ],
}


def remover(char):
    if char in const.farsi:
        return char
    if char in const.alef:
        return const.alef[0]
    if char in const.vav:
        return const.vav[0]
    if char in const.heh:
        return const.heh[0]
    if char in const.yah:
        return const.yah[0]
    if char in const.kaf:
        return const.kaf[0]
    return ''


def pre_process(text):
    persian_words = map(remover, text)
    sentence = ''.join(persian_words)
    if (len(sentence) < 20):
      return None
    normal_sentence = (normalizer, sentence)
    word_tokens = word_tokenize(sentence)

    for w in word_tokens:
      if w in swearing_words:
        return None

    filtered_stopwords = [w for w in word_tokens if w not in stopwords and len(w) > 1]

    if (len(filtered_stopwords) < 5):
      return None
    filtered_stopwords = ' '.join(filtered_stopwords)
    return filtered_stopwords


def main(args):
    results = {}
    for topic in hashtags.keys():
        scraper = Twitter_scraper(
            max_results=args.max_results,
            hashtags=hashtags[topic],
            lang=args.fa,
            until=args.until,
            since=args.since,
            with_replies=args.with_replies,
        )
        results[topic] = scraper.basic_mode()
    df = pd.Dataframe(results)

    # preprocess
    df = df[df['username'].notna()]
    tweets = map(pre_process, df.text)
    tweets = list(tweets)
    df['processed_text'] = tweets
    df = df[df['processed_text'].notna()]
    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset='tweet_id')
    print('-- Dataframe shape: {}'.format(df.shape))
    df = df.groupby('topic').apply(lambda x: x.sample( len(x) if len(x) < 10000 else 10000)).reset_index(drop=True)
    df = df.reset_index(drop=True)

    df.to_csv(f"../datasets/twitter_dataset.csv", index=False)
    print('[ OK ] Dataset created.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_results", default=(2 * (10 ** 4)), type=int)
    parser.add_argument("--lang", default="fa", type=str)
    parser.add_argument("--until", default="2022-02-10", type=str)
    parser.add_argument("--since", default="2019-06-01", type=str)
    parser.add_argument("--with_replies", default=False, type=bool)
    args = parser.parse_args()
    print(args)
    main(args)
