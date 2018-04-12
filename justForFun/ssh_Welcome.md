## Welcome message for ssh login
Create or edit existing file `~/.ssh/rc` with below contents:

```bash
#!/bin/bash
fortune -s | cowsay -f $(ls /usr/share/cowsay/cows/eyes.cow)
echo "$(uptime)"
echo ""
```

I've set `eyes.cow` as my default ASCII pictures, you can change it with whichever you like, or better, let it take a random one by using:  
```bash
fortune | cowsay -f $(ls /usr/share/cowsay/cows/ | shuf -n 1)
```
