import sys

def generate_nginx_conf(allowed_ips):
    allowed_ips_block = "\n".join(f"allow {ip};" for ip in allowed_ips)
    conf_template = f"""
server {{
    listen 8080;
    server_name localhost;

    location / {{
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        {allowed_ips_block}
        deny all;
    }}
}}
"""
    with open("/app/nginx.conf", "w") as f:
        f.write(conf_template)
    
    print("nginx.conf generated successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_nginx_conf.py <ip1> <ip2> ... <ipN>")
    else:
        generate_nginx_conf(sys.argv[1:])
