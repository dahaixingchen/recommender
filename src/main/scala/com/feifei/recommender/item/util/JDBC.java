package com.feifei.recommender.item.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

public class JDBC {
    public static void main(String[] args) throws Exception {

        Class.forName("com.mysql.jdbc.Driver");

        String url = "jdbc:mysql://node01:3306/program_db";
        String username = "root";
        String password = "123123";
        Connection conn = null;

        conn = DriverManager.getConnection(url, username, password);


        String pathname = "D:\\点播数据\\item.csv";

        FileReader reader = new FileReader(pathname);
        BufferedReader br = new BufferedReader(reader);
        String line;
        String insertSQL = "";
        Statement st = conn.createStatement();
        while ((line = br.readLine()) != null) {
            int dindex = line.lastIndexOf(",");
            String is_3d = line.substring(dindex + 1, line.length());
            line = line.substring(0, dindex);


            dindex = line.lastIndexOf(",");
            String quality = line.substring(dindex + 1, line.length());
            line = line.substring(0, dindex);

            dindex = line.lastIndexOf(",");
            String language = line.substring(dindex + 1, line.length());
            line = line.substring(0, dindex);

            dindex = line.lastIndexOf(",");
            String area = line.substring(dindex + 1, line.length());
            line = line.substring(0, dindex);


            dindex = line.lastIndexOf(",");
            String content_model = line.substring(dindex + 1, line.length()).replace("\"","'");
            line = line.substring(0, dindex);

            dindex = line.lastIndexOf(",");
            String length = line.substring(dindex + 1, line.length());
            line = line.substring(0, dindex);


            dindex = line.lastIndexOf(",");
            String foucs = line.substring(dindex + 1, line.length()).replace("\"","'");
            line = line.substring(0, dindex);


            dindex = line.lastIndexOf(",");
            String keywords = line.substring(dindex + 1, line.length()).replace("\"","'");
            line = line.substring(0, dindex);



            int index = line.indexOf(",");
            String id = line.substring(0, index);
            line = line.substring(index + 1);

            index = line.indexOf(",");
            String create_date = line.substring(0, index);
            line = line.substring(index + 1);

            index = line.indexOf(",");
            String air_date = line.substring(0, index);
            line = line.substring(index + 1);


            index = line.indexOf(",");
            String title = line.substring(0, index).replace("\"","'");
            line = line.substring(index + 1);

            index = line.indexOf(",");
            String name = line.substring(0, index).replace("\"","");
            line = line.substring(index + 1);

            line = line.replace("\"","");


            insertSQL = "insert into item_info values("+id +",\""+create_date+"\",\""+air_date+"\",\""+title+"\",\""+name+"\",\""+line+"\",\""+keywords+"\",\""+foucs+"\",\""+length+"\",\""+content_model+"\",\""+area+"\",\""+language+"\",\""+quality+"\",\""+is_3d+"\",null)";
            System.out.println(insertSQL);
            System.out.println(line);
            int resultSet = st.executeUpdate(insertSQL);
            if(resultSet>0){
                System.out.println("success");
            }
        }
        //4.向数据库发sql
        String sql = "select * from item_info";
    }
}
