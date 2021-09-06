import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType, DateType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    This initiates the spark session, names it, and configures the session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    This section reads in the data, creates the tables from the song data files and convert to parquet format.
    """
    # get filepath to song data file
    song_data = "./data/songs/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 
                             'title', 
                             'artist_id', 
                             'year', 
                             'duration']).dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data+"songs/songs_table.parquet")

    # extract columns to create artists table
    artists_table = df.selectExpr(['artist_id', 
                                   'artist_name as name', 
                                   'artist_location as location', 
                                   'artist_latitude as latitude', 
                                   'artist_longitude as longitude']).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data+"artists/artists_table.parquet")


def process_log_data(spark, input_data, output_data):
    """
    This section reads in the data, creates the tables from the log data files and convert to parquet format.
    """
    # get filepath to log data file
    log_data = "./data/logs/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df[df['page']=='NextSong']

    # extract columns for users table    
    users_table = df.selectExpr(['userId as user_id', 
                                 'firstName as first_name', 
                                 'lastName as last_name', 
                                 'gender', 
                                 'level']).dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(output_data+"users/users_table.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(int(int(x) / 1000)), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime(int(x)),DateType())
    df = df.withColumn("datetime", get_timestamp(df.ts))
    
    # extract columns to create time table
    time_table = df.select('ts','datetime','timestamp',
                           year(df.datetime).alias('year'),
                           month(df.datetime).alias('month')
                          ).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data+"time/time_table.parquet")

    # read in song data to use for songplays table
    song_data = "./data/songs/*.json"
    song_df = spark.read.json(song_data)
    song_df.createOrReplaceTempView("song_df")
    log_data = "./data/logs/*.json"
    log_df = spark.read.json(log_data)
    log_df.createOrReplaceTempView("log_df")
    

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""SELECT itemInSession as songplay_id,
                                           l.ts as start_time,
                                           l.userId as user_id,
                                           l.level,
                                           s.song_id,
                                           s.artist_id,
                                           l.sessionId as session_id,
                                           l.location,
                                           l.userAgent as user_agent,
                                           year(df.datetime).alias('year'),
                                           month(df.datetime).alias('month')
                                           FROM log_df l
                                           JOIN song_df s ON l.artist = s.artist_name and l.song = s.title""") 

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode('overwrite').parquet(output_data+"songplays/songplays_table.parquet")


def main():
    """
    Creates the spark session and leverages the source data s3 bucket to process the functions above and place it in a created s3 folder for the project.
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
