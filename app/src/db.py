import psycopg2
import os
import pandas as pd
from db_names import *
def connect():
	return psycopg2.connect(
	    host = os.environ['AWS_RDS_ENDPOINT'],
	    port = 5432,
	    user = os.environ['AWS_RDS_USER'],
	    password = os.environ['AWS_RDS_PASS'],
	    database='windsitedb'
	    )

def sql_request(sql,connection):
	return pd.read_sql(sql, con=connection)

def load_all_results(connection):
	sql = """
		SELECT *
		FROM "results-21" t
	"""
	return sql_request(sql,connection)

def load_with_constraints(connection,max_trans,max_road,min_res):
	sql = """
		SELECT *
		FROM "results-21" t
		WHERE """ \
		+ TRANS_DIST_COL + ' <= ' + str(max_trans) + ' AND ' \
		+ ROAD_DIST_COL + ' <= ' + str(max_road) + ' AND ' \
		+ RES_ROAD_DIST_COL + ' >= ' + str(min_res)
	return sql_request(sql,connection)