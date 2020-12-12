import psycopg2
import datetime

from data_import.aisimport import AISImporterPostgres
from aisdata import scope
from data_preprocessing import preprocessing

importer = AISImporterPostgres('navi.ka-projekte.corp', 'ais', 'blejoh', 'blejoh')
test_ships = importer.import_data((1547301493,1547311493), (50.0, 51.0), (0.0, 0.5))
test_scope = scope.Scope(0.0, 1.0, 48, 59, 0, 1500000000)
test_preprocessing = preprocessing.Preprocessing(test_scope, test_ships)
test_preprocessing.filter_by_scope()

#try:
#    connect_str = "dbname='ais' user='blejoh' host='navi.ka-projekte.corp' " + \
#                  "password='blejoh'"
#    # use our connection values to establish a connection
#    conn = psycopg2.connect(connect_str)
#    # create a psycopg2 cursor that can execute queries
#    cursor = conn.cursor()
#    # create a new table with a single column called "name"
#    #cursor.execute("""CREATE TABLE tutorials (name char(40));""")
#    # run a SELECT statement - no data in there, but we can try it
#    #cursor.execute("""SELECT * from public.vessels WHERE shipname = 'JAXON AARON'""")
#    #cursor.execute("""SELECT * from public.positions WHERE mmsi = '367599780'""")
#    cursor.execute("""SELECT * from public.vessels WHERE lastseen = timestamp without time zone '2019-01-17 01:00' + interval '1 day'""")
#    #date = datetime.datetime(2019, 1, 17, 20, 12, 47, 843484)
#    #cursor.mogrify("SELECT * from public.vessels WHERE lastseen = timestamp %s ", 
#    #        (date.date()))
#    rows = cursor.fetchall()
#    print(rows)
#except Exception as e:
#    print("Uh oh, can't connect. Invalid dbname, user or password?")
#    print(e)

# Useful queries:
# SELECT p.mmsi, p.timedate, v.shipname, MIN(p.timedate - d.timedate) AS timediff, ST_X(p.position) as lon,
# ST_Y(p.position), d.destination as lat, shiptype FROM public.vessels v, public.positions p, public.destinations d
# WHERE p.mmsi = v.mmsi and p.mmsi = d.mmsi and p.timedate between (to_timestamp(1548401493) AT TIME ZONE 'UTC') and
# (to_timestamp(1548411423) AT TIME ZONE 'UTC') and  position && ST_MakeEnvelope(0.0, 50.0, 0.5, 51) and v.shiptype = 70
# GROUP BY p.mmsi, p.timedate, v.shipname, v.shiptype, p.position, d.destination order by p.mmsi, p.timedate, timediff;

#SELECT name.mmsi, name.time, name.shipname, MIN(timediff) 
#FROM (
#    SELECT p.mmsi, p.timedate as time, v.shipname, MIN(p.timedate - d.timedate) AS timediff, ST_X(p.position) as lon, ST_Y(p.position), d.destination as lat, shiptype 
#    FROM public.vessels v, public.positions p INNER JOIN public.destinations d ON p.mmsi = d.mmsi 
#    WHERE p.mmsi = v.mmsi and p.timedate between (to_timestamp(1548401493) AT TIME ZONE 'UTC') and (to_timestamp(1548411423) AT TIME ZONE 'UTC') and  position && ST_MakeEnvelope(0.0, 50.0, 0.5, 51) and v.shiptype = 70 
#    GROUP BY p.mmsi, p.timedate, v.shipname, v.shiptype, p.position, d.destination 
#    order by p.mmsi, p.timedate, timediff
#    ) AS name 
#GROUP BY name.mmsi, name.time, name.shipname;
#
#
#SELECT name.mmsi, name.timedate, name.speed, name.course, name.heading, name.lon, name.lat,
#        name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught,
#        name.destination, name.eta, MIN(timediff)
#FROM (
#    SELECT p.mmsi, p.timedate, p.speed, p.course, p.heading, ST_X(p.position) as lon, ST_Y(p.position) as lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught, d.destination, d.eta,
#        MIN(p.timedate - d.timedate) AS timediff
#    FROM public.vessels v, public.positions p 
#    INNER JOIN public.destinations d ON p.mmsi = d.mmsi 
#    WHERE p.mmsi = v.mmsi 
#        and p.timedate between (to_timestamp(1548401493) AT TIME ZONE 'UTC') and (to_timestamp(1548411423) AT TIME ZONE 'UTC') 
#        and  position && ST_MakeEnvelope(0.0, 50.0, 0.5, 51) 
#        and v.shiptype = 70 
#    GROUP BY p.mmsi, p.timedate, p.speed, p.course, p.heading, lon, lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught, d.destination, d.eta
#    ORDER BY p.mmsi, p.timedate, timediff
#    ) 
#AS name
#    GROUP BY name.mmsi, name.timedate, name.speed, name.course, name.heading, name.lon, name.lat,
#        name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught,
#        name.destination, name.eta;
#
#
#SELECT name.mmsi, name.timedate, name.lon, name.lat, name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta, MIN(name.timediff)
#FROM (
#    SELECT p.mmsi, p.timedate, p.speed, p.course, p.heading, ST_X(p.position) as lon, ST_Y(p.position) as lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught,
#        MIN(ABS(p.timedate - d.timedate)) AS timediff
#    FROM public.vessels v, public.positions p 
#    INNER JOIN public.destinations d ON p.mmsi = d.mmsi 
#    WHERE p.mmsi = v.mmsi 
#        and p.timedate between (to_timestamp(1548401493) AT TIME ZONE 'UTC') and (to_timestamp(1548411423) AT TIME ZONE 'UTC') 
#        and  position && ST_MakeEnvelope(0.0, 50.0, 0.5, 51) 
#    GROUP BY p.mmsi, p.timedate, p.speed, p.course, p.heading, lon, lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught
#    ORDER BY p.mmsi, p.timedate, timediff
#    ) 
#AS name, public.destinations dest WHERE name.mmsi = dest.mmsi and name.timedate = dest.timedate + name.timediff
#    GROUP BY name.mmsi, name.timedate, name.speed, name.course, name.heading, name.lon, name.lat,
#        name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta;
#
#
#
#
#SELECT name.mmsi, name.timedate, name.lon, name.lat, name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta, MIN(name.timediff)
#FROM (
#    SELECT p.mmsi, p.timedate, p.speed, p.course, p.heading, ST_X(p.position) as lon, ST_Y(p.position) as lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught,
#        MIN(p.timedate - d.timedate) AS timediff
#    FROM public.vessels v, public.positions p 
#    INNER JOIN public.destinations d ON p.mmsi = d.mmsi 
#    WHERE p.mmsi = v.mmsi 
#        and p.timedate between (to_timestamp(1547301493) AT TIME ZONE 'UTC') and (to_timestamp(1547311423) AT TIME ZONE 'UTC') 
#        and  position && ST_MakeEnvelope(0.0, 50.0, 0.5, 51) and p.timedate > d.timedate
#    GROUP BY p.mmsi, p.timedate, p.speed, p.course, p.heading, lon, lat,
#        v.shipname, v.shiptype, v.to_bow, v.to_stern, v.to_starboard, v.to_port, v.draught
#    ORDER BY p.mmsi, p.timedate, timediff
#    ) 
#AS name, public.destinations dest WHERE name.mmsi = dest.mmsi and name.timedate = dest.timedate + name.timediff
#    GROUP BY name.mmsi, name.timedate, name.speed, name.course, name.heading, name.lon, name.lat,
#        name.shipname, name.shiptype, name.to_bow, name.to_stern, name.to_starboard, name.to_port, name.draught, dest.destination, dest.eta 
#    ORDER BY name.mmsi, name.timedate;
