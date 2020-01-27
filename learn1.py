import pandas as pd
from sqlalchemy import create_engine
import cx_Oracle
db= cx_Oracle.connect('myils','abc4R5T6y','112.80.18.150:1522/book_st')
cr=db.cursor()
sql = 'select ADD_TIME,ORDERNO ,LOCAL_CODE from SELL_BOOK_ORDER@sdx_online where ADD_TIME > TO_DATE(\'2019-11-01\',\'yyyy-mm-dd\') AND ADD_TIME < TO_DATE(\'2019-12-01\',\'yyyy-mm-dd\') order by ADD_TIME asc'
cr.execute(sql)
rs = cr.fetchall()
zz=pd.DataFrame(rs)
db.close()