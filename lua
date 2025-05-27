http {
    lua_package_path "/appbin/lua-resty-jwt/lib/?.lua;;";
    lua_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;

    server {
        listen 443 ssl;
        server_name chat-qa.app.xyz.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location /ws/ {
            access_by_lua_block {
                local jwt = require "resty.jwt"
                local cjson = require "cjson"
                local ngx_decode_base64 = ngx.decode_base64

                -- 1. Extract Authorization header
                local auth_header = ngx.var.http_authorization
                if not auth_header or not auth_header:find("Bearer ") then
                    return ngx.exit(401)
                end
                local token = auth_header:match("Bearer%s+(.+)")

                -- 2. Load ES256 public key
                local public_key = [[
-----BEGIN PUBLIC KEY-----
YOUR_ES256_PUBLIC_KEY_HERE
-----END PUBLIC KEY-----
]]

                -- 3. Validate JWT
                local jwt_obj = jwt:verify(public_key, token, {
                    alg = "ES256"
                })

                if not jwt_obj.verified then
                    ngx.log(ngx.ERR, "JWT validation failed: ", jwt_obj.reason)
                    return ngx.exit(401)
                end

                -- Optional: log claims or validate `exp`, `aud`, etc.
                ngx.log(ngx.INFO, "JWT valid. Subject: ", jwt_obj.payload.sub)
            }

            # Allow WebSocket upgrade
            proxy_pass http://backend_ws;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Upstream to your backend WebSocket server
        upstream backend_ws {
            server 127.0.0.1:8001;
        }
    }
}
