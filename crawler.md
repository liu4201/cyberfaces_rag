## This is an instruction to permit external access in the dev/prod container.
```
mysql -P 3306 -u root -p cybertraining

CREATE USER 'crawler'@'149.165.153.23' IDENTIFIED BY '<PASSWORD>';
GRANT SELECT ON cybertraining.* TO 'crawler'@'149.165.153.23';
REVOKE INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, INDEX, EXECUTE
ON cybertraining.* FROM 'crawler'@'149.165.153.23';
FLUSH PRIVILEGES;


SELECT User, Host, plugin From user;
SHOW GRANTS FOR 'crawler'@'149.165.153.23';
```