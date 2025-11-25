# Manual Integration Guide
# Step-by-Step Instructions for Adding SpikingBrain to Living Intelligence

This guide walks you through integrating SpikingBrain-7B content into your
living-intelligence website's "Explore Technology" menu.

---

## 🎯 Goal

Add a new menu path:
```
Home → Explore → Technology → SpikingBrain-7B
```

With comprehensive content showcasing the neuromorphic AI technology.

---

## 📋 Step-by-Step Process

### Step 1: Locate Your living-intelligence Repository

```bash
# Navigate to your repository
cd /path/to/living-intelligence

# Verify it's the correct repo
ls -la
# You should see your website files
```

### Step 2: Identify Your Directory Structure

Run this command to see your structure:

```bash
tree -L 2 -d
# Or if tree is not installed:
find . -maxdepth 2 -type d | sort
```

**Common structures:**

**Option A: docs/ based**
```
living-intelligence/
├── docs/
│   ├── index.md
│   ├── about.md
│   └── explore/          ← We'll add here
```

**Option B: content/ based**
```
living-intelligence/
├── content/
│   ├── _index.md
│   └── explore/          ← We'll add here
```

**Option C: src/pages/ based**
```
living-intelligence/
├── src/
│   └── pages/
│       ├── index.md
│       └── explore/      ← We'll add here
```

**📝 Note your structure:** _________________________

### Step 3: Create Directory Structure

Based on your structure from Step 2:

```bash
# If using docs/:
mkdir -p docs/explore/technology

# If using content/:
mkdir -p content/explore/technology

# If using src/pages/:
mkdir -p src/pages/explore/technology
```

### Step 4: Copy Content Files

```bash
# Set your content directory variable
CONTENT_DIR="docs"  # or "content" or "src/pages"

# Copy the main SpikingBrain page
cp /path/to/SpikingBrain-7B/living-intelligence-integration/explore-spikingbrain.md \
   $CONTENT_DIR/explore/technology/spikingbrain.md

# Copy the technology index page
cp /path/to/SpikingBrain-7B/living-intelligence-integration/technology-index.md \
   $CONTENT_DIR/explore/technology/index.md

# Verify files were copied
ls -la $CONTENT_DIR/explore/technology/
```

**Expected output:**
```
-rw-r--r-- 1 user user 14060 Nov 25 10:00 spikingbrain.md
-rw-r--r-- 1 user user  8647 Nov 25 10:00 index.md
```

### Step 5: Update Navigation Configuration

Find your navigation/menu config file:

```bash
# Common locations:
ls -la _config.yml                    # Jekyll
ls -la config.toml                    # Hugo (TOML)
ls -la config.yaml                    # Hugo (YAML)
ls -la mkdocs.yml                     # MkDocs
ls -la .vuepress/config.js            # VuePress
ls -la docusaurus.config.js           # Docusaurus
ls -la gatsby-config.js               # Gatsby
ls -la next.config.js                 # Next.js
```

**📝 Your config file:** _________________________

Now open it and add the navigation entry:

```bash
# Open in your editor
vim _config.yml  # or nano, code, etc.
```

**Add navigation entry** (choose based on your platform):

<details>
<summary><b>For Jekyll (_config.yml)</b></summary>

```yaml
# Find the nav or navigation section and add:
nav:
  - title: "Explore"
    submenu:
      - title: "Technology"
        url: "/explore/technology/"
        submenu:
          - title: "SpikingBrain-7B"
            url: "/explore/technology/spikingbrain"
```
</details>

<details>
<summary><b>For Hugo (config.toml)</b></summary>

```toml
# Add to [[menu.main]] section:
[[menu.main]]
  name = "Explore"
  url = "/explore/"
  weight = 2

[[menu.main]]
  parent = "Explore"
  name = "Technology"
  url = "/explore/technology/"

[[menu.main]]
  parent = "Technology"
  name = "SpikingBrain-7B"
  url = "/explore/technology/spikingbrain/"
```
</details>

<details>
<summary><b>For MkDocs (mkdocs.yml)</b></summary>

```yaml
# In the nav section:
nav:
  - Home: index.md
  - Explore:
    - Technology:
      - Overview: explore/technology/index.md
      - SpikingBrain-7B: explore/technology/spikingbrain.md
```
</details>

<details>
<summary><b>For VuePress (.vuepress/config.js)</b></summary>

```javascript
// In themeConfig.nav:
{
  text: 'Explore',
  items: [
    {
      text: 'Technology',
      items: [
        { text: 'SpikingBrain-7B', link: '/explore/technology/spikingbrain' }
      ]
    }
  ]
}
```
</details>

**See `navigation-config-samples.yml` for ALL platforms!**

### Step 6: Test Locally

Build and serve your site locally:

```bash
# Jekyll
bundle exec jekyll serve

# Hugo
hugo server

# MkDocs
mkdocs serve

# VuePress
npm run docs:dev

# Docusaurus
npm start

# Next.js
npm run dev
```

**Visit your local site:**
```
http://localhost:XXXX/explore/technology/spikingbrain
```

**Test checklist:**
- [ ] Page loads correctly
- [ ] Navigation menu shows "Explore → Technology → SpikingBrain-7B"
- [ ] All internal links work
- [ ] External links work (GitHub, ModelScope, etc.)
- [ ] Code blocks are formatted correctly
- [ ] Tables render properly
- [ ] Mobile view looks good

### Step 7: Commit Changes

```bash
# Check what changed
git status

# Add the new files
git add docs/explore/technology/  # or content/explore/ or src/pages/explore/
git add _config.yml  # or your nav config file

# Commit
git commit -m "Add SpikingBrain-7B to Explore Technology section

- Add comprehensive SpikingBrain-7B technology page
- Add technology section index page
- Update navigation to include new content
- Includes architecture diagrams, demos, and integration guides
"

# Push to your repository
git push origin main  # or your branch name
```

### Step 8: Deploy to Production

Depending on your hosting:

**GitHub Pages:**
```bash
# Usually auto-deploys after push to main
# Check: https://github.com/Lightiam/living-intelligence/actions
```

**Netlify:**
```bash
# Auto-deploys from GitHub
# Or manual: netlify deploy --prod
```

**Vercel:**
```bash
# Auto-deploys from GitHub
# Or manual: vercel --prod
```

**Custom Server:**
```bash
# SSH to server and pull changes
ssh user@your-server.com
cd /var/www/living-intelligence
git pull
# Rebuild if needed
```

### Step 9: Verify Production Deployment

Visit your live site:
```
https://your-site.com/explore/technology/spikingbrain
```

**Final checklist:**
- [ ] Page is live and accessible
- [ ] Navigation menu shows correctly
- [ ] All links work
- [ ] Images load (if any)
- [ ] Mobile responsive
- [ ] Page loads quickly (< 3 seconds)

### Step 10: Announce!

Share your new technology page:
- Social media
- Newsletter
- Blog post
- Community forums

---

## 🔄 Alternative: Use Automated Script

If you prefer automation:

```bash
# Copy integration files to your living-intelligence repo
cd /path/to/living-intelligence

# Run the auto-integration script
python3 /path/to/SpikingBrain-7B/living-intelligence-integration/auto_integrate.py .

# Follow the prompts
```

The script will:
- Auto-detect your site generator
- Find content directory
- Copy files to correct locations
- Suggest navigation updates
- Create integration report

---

## 🐛 Troubleshooting

### Issue: "Page not found" after integration

**Cause:** URL path doesn't match file location

**Solution:**
```bash
# Check file location
ls -la docs/explore/technology/spikingbrain.md

# Check navigation URL matches
# Should be: /explore/technology/spikingbrain
#            (without .md extension usually)
```

### Issue: Navigation menu doesn't show new item

**Cause:** Site hasn't been rebuilt, or config syntax error

**Solution:**
```bash
# Rebuild the site
jekyll build  # or hugo, mkdocs build, etc.

# Check for config errors
jekyll serve --trace  # Shows detailed errors

# Verify config syntax
yamllint _config.yml  # For YAML files
```

### Issue: Links are broken

**Cause:** Relative vs. absolute path issues

**Solution:**
```markdown
<!-- Use absolute paths from root -->
[Link](/explore/technology/spikingbrain)

<!-- Not relative -->
[Link](../technology/spikingbrain)  # Might break
```

### Issue: Styling looks off

**Cause:** Need custom CSS for code blocks or tables

**Solution:**
```css
/* Add to your site's CSS */
.technology-page table {
  width: 100%;
  border-collapse: collapse;
}

.technology-page pre {
  background: #f5f5f5;
  padding: 1rem;
  border-radius: 4px;
}
```

---

## 📞 Need Help?

### Can't figure out your site structure?

Share the output of:
```bash
cd /path/to/living-intelligence
tree -L 2
# Or:
find . -maxdepth 2 -type f -name "*.md" -o -name "*.yml" -o -name "*.toml" -o -name "*.js"
```

### Can't find navigation config?

Look for these patterns:
```bash
find . -name "*config*" -type f
find . -name "*navigation*" -type f
find . -name "SUMMARY.md"  # GitBook
```

### Integration not working?

Check:
1. File paths are correct
2. Navigation syntax matches your platform
3. Site has been rebuilt
4. No syntax errors in config
5. Server restarted (if needed)

---

## ✅ Success Criteria

You'll know integration is successful when:

1. ✅ You can navigate to the page via menu
2. ✅ All sections render correctly
3. ✅ Code blocks are formatted
4. ✅ Links work (internal and external)
5. ✅ Page is mobile responsive
6. ✅ Loads in < 3 seconds

---

## 📚 Reference Files

All integration materials are in:
```
living-intelligence-integration/
├── README.md                          # Package overview
├── INTEGRATION_INSTRUCTIONS.md        # Detailed guide
├── MANUAL_INTEGRATION_GUIDE.md        # This file
├── explore-spikingbrain.md            # Main content
├── technology-index.md                # Index page
├── navigation-config-samples.yml      # Config samples
└── auto_integrate.py                  # Automation script
```

---

## 🎓 Platform-Specific Tips

### Jekyll
- Frontmatter required in all .md files
- Use `permalink` for custom URLs
- Rebuild with `bundle exec jekyll build`

### Hugo
- Use `index.md` for section pages
- Check `baseURL` in config.toml
- Archetypes can help with frontmatter

### MkDocs
- Navigation order matters in mkdocs.yml
- Use `site_url` for absolute links
- Material theme has extra features

### VuePress
- Create `.vuepress/config.js` if missing
- Use frontmatter for page metadata
- Hot reload should work automatically

---

## 🎬 Final Step

Once integrated, test this user journey:

1. User visits your homepage
2. Clicks "Explore" in menu
3. Clicks "Technology"
4. Sees "SpikingBrain-7B" option
5. Clicks it
6. Reads comprehensive page
7. Clicks "Run Demo" button
8. Gets engaged with the technology!

---

**Good luck with your integration!** 🚀

If you get stuck, refer to:
- `INTEGRATION_INSTRUCTIONS.md` for detailed help
- `navigation-config-samples.yml` for exact configs
- SpikingBrain-7B repo for source material
