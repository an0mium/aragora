"""
Migration versions package.

Each migration is in a separate file named v{version}_{name}.py
with a `migration` variable containing the Migration definition.

Example:
    # v20240115120000_add_users.py
    from aragora.migrations import Migration

    migration = Migration(
        version=20240115120000,
        name="Add users table",
        up_sql="CREATE TABLE users (...);",
        down_sql="DROP TABLE users;",
    )
"""
