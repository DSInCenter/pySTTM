# https://github.com/JustAnotherArchivist/snscrape/blob/master/snscrape/modules/twitter.py
import snscrape.modules.twitter as sntwitter
import pandas as pd
from random import random
from datetime import date
from multiprocessing.dummy import Pool
import time


class TwitterScraper:
    def __init__(
        self,
        max_results: int,
        all_words=[],
        exact_pharase=[],
        any_words=[],
        none_words=[],
        hashtags=[],
        mentioned_users=[],
        from_users=[],
        to_users=[],
        with_links=True,
        with_replies=True,
        **kwargs,
    ):
        """
        :param max_results: Number of tweets will be captured per user.
        :param all_words: ['what’s', 'happening'] · contains both “what’s” and “happening”
        :param exact_pharase: ['happy hour'] · contains the exact phrase “happy hour”
        :param any_words: ['cats', 'dogs'] · contains either “cats” or “dogs” (or both)
        :param none_words: ['cats', 'dogs'] · does not contain “cats” and does not contain “dogs”
        :param hashtags: ['#ThrowbackThursday'] or ['ThrowbackThursday'] · contains the hashtag #ThrowbackThursday
        :param mentioned_users: ['@SFBART', '@Caltrain'] or ['SFBART', 'Caltrain'] · mentions @SFBART or mentions @Caltrain
        :param from_users: ['@Twitter'] or ['Twitter'] · sent from @Twitter
        :param to_users: ['@Twitter'] or ['Twitter'] · sent in reply to @Twitter
        :param with_links: Tweets contain link
        :param with_replies: Tweets as a replies
        :param kwargs: The setting of start_time, end_time, and language of the tweets

        """

        self.number_of_user = 0
        self.max_results = max_results
        self.all_words = TwitterScraper.all_of_these_words(all_words)
        self.exact_pharase = TwitterScraper.any_of_these_exact_pharase(exact_pharase)
        self.any_words = TwitterScraper.any_of_these_words(any_words)
        self.none_words = TwitterScraper.none_of_these_words(none_words)
        self.these_hashtags = TwitterScraper.any_of_these_hashtags(hashtags)
        self.mentioned_users = TwitterScraper.mentioning_these_users(mentioned_users)
        self.with_links = f"-filter:links" if not with_links else ""
        self.with_replies = f"-filter:replies" if not with_replies else ""

        self.query_dict = {
            "all_words": self.all_words,
            "exact_pharase": self.exact_pharase,
            "any_words": self.any_words,
            "none_words": self.none_words,
            "these_hashtags": self.these_hashtags,
            "mentioned_users": self.mentioned_users,
            "with_links": self.with_links,
            "with_replies": self.with_replies,
        }

        self.query_dict["from"] = TwitterScraper.f_or_t_users(from_users, "from")
        self.query_dict["to"] = TwitterScraper.f_or_t_users(to_users, "to")

        for key, value in kwargs.items():
            self.query_dict[key] = f"({key}:{value})"

    @staticmethod
    def f_or_t_users(users, key):
        if not users:
            return ""
        tmp_list = [f"{key}:{user}" for user in users]
        return "(" + " OR ".join(tmp_list) + ")"

    @staticmethod
    def all_of_these_words(all_words):
        if not all_words:
            return ""
        return " ".join(all_words)

    @staticmethod
    def any_of_these_words(any_words):
        if not any_words:
            return ""
        return "(" + " OR ".join(any_words) + ")"

    @staticmethod
    def any_of_these_exact_pharase(exact_pharase):
        if not exact_pharase:
            return ""
        return '("' + '" OR "'.join(exact_pharase) + '")'

    @staticmethod
    def none_of_these_words(none_words):
        if not none_words:
            return ""
        return "-" + " -".join(none_words)

    @staticmethod
    def any_of_these_hashtags(hashtags):
        if not hashtags:
            return ""
        tmp_list = ["#" + h.replace("#", "") for h in hashtags]
        return "(" + " OR ".join(tmp_list) + ")"

    @staticmethod
    def mentioning_these_users(users):
        if not users:
            return ""
        tmp_list = ["@" + h.replace("@", "") for h in users]
        return "(" + " OR ".join(tmp_list) + ")"

    def create_query(self, query_dict):
        tmp_string = ""
        res = dict([(key, val) for key, val in query_dict.items() if val])
        del query_dict
        query = " ".join(res.values())
        del res

        return query

    def crawler(self, query, error_counter=0):
        """
        This is the main function to send the query to Twitter and collect the tweets.

        :param query: The Twitter query that we built it.
        """
        # Creating list to append tweet data
        tweets_list = []
        try:
            # Using TwitterSearchScraper to scrape data and append tweets to list
            scraper = sntwitter.TwitterSearchScraper(query)
            i = 0
            for tweet in scraper.get_items():  # declare a username
                if i >= self.max_results:  # check number and date
                    break

                tweets_list.append(
                    [
                        tweet.date,
                        tweet.id,
                        tweet.content,
                        tweet.replyCount,
                        tweet.retweetCount,
                        tweet.likeCount,
                        tweet.user.username,
                        tweet.lang,
                        tweet.media,
                        tweet.hashtags,
                    ]
                )  # declare the attributes to be returned
                i += 1
        except Exception as e:
            if "Unable to find guest token" in str(e):
                error_counter += 1
                if error_counter > 3:
                    error_counter = 0
                    print("Sleep Time!")
                    time.sleep(30.3 * 60)
                    print("Morning!")

                return self.crawler(query, error_counter)
            print(f"query: {query} , {e}")

        # Creating a dataframe from the tweets list above
        tweets_df = pd.DataFrame(
            tweets_list,
            columns=[
                "datetime",
                "tweet_id",
                "text",
                "reply_count",
                "retweet_count",
                "like_count",
                "username",
                "lang",
                "media",
                "hashtags",
            ],
        )
        return tweets_df

    def basic_mode(self):
        query = self.create_query(self.query_dict)
        return self.crawler(query)

    def user_crawler(self, user):
        """
        Calling the query function for a specefic user.

        :param user: Account username in Twitter
        """
        tmp_dict = self.query_dict.copy()
        tmp_dict["from"] = f"(from:{user})"
        query = self.create_query(tmp_dict)
        del tmp_dict
        return self.crawler(query)

    def user_mode(self, user_list):
        """
        parallelize the process with multithreading(call query function per user)

        :param user_list: List of users that we going to collect their tweets.
        """
        user_crawler = self.user_crawler
        pool = Pool(22)
        df_list = pool.map(user_crawler, user_list)
        pool.close()
        pool.join()
        result_df = pd.concat(df_list, ignore_index=True)
        return result_df
