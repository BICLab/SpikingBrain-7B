# Living Intelligence Integration Instructions

This document explains how to integrate the SpikingBrain-7B content into your **living-intelligence** repository's "Explore Technology" section.

---

## 📁 Files Included

```
living-intelligence-integration/
├── INTEGRATION_INSTRUCTIONS.md    # This file
├── explore-spikingbrain.md         # Main technology page
├── navigation-config.yml           # Menu/navigation structure
├── technology-index.md             # Index page for explore section
└── assets/                         # Visual assets (if needed)
```

---

## 🚀 Quick Integration (3 Steps)

### Step 1: Copy Files to Your Repository

Assuming your living-intelligence repo has a structure like:

```
living-intelligence/
├── docs/
│   ├── explore/
│   │   └── technology/         ← Add files here
│   └── index.md
├── src/
└── README.md
```

**Commands:**

```bash
# Navigate to your living-intelligence repo
cd /path/to/living-intelligence

# Create explore/technology directory if it doesn't exist
mkdir -p docs/explore/technology

# Copy the SpikingBrain page
cp /path/to/SpikingBrain-7B/living-intelligence-integration/explore-spikingbrain.md \
   docs/explore/technology/spikingbrain.md

# Copy the index page
cp /path/to/SpikingBrain-7B/living-intelligence-integration/technology-index.md \
   docs/explore/technology/index.md
```

### Step 2: Update Navigation/Menu

The exact steps depend on your site generator (Jekyll, Hugo, VuePress, etc.):

#### If using Jekyll (_config.yml):

```yaml
nav:
  - title: Home
    url: /
  - title: Explore
    url: /explore/
    submenu:
      - title: Technology
        url: /explore/technology/
        submenu:
          - title: SpikingBrain-7B
            url: /explore/technology/spikingbrain
```

#### If using Hugo (config.toml):

```toml
[[menu.main]]
  name = "Explore"
  url = "/explore/"
  weight = 2

[[menu.main]]
  parent = "Explore"
  name = "Technology"
  url = "/explore/technology/"
  weight = 1

[[menu.main]]
  parent = "Technology"
  name = "SpikingBrain-7B"
  url = "/explore/technology/spikingbrain/"
  weight = 1
```

#### If using VuePress (.vuepress/config.js):

```javascript
module.exports = {
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
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
    ]
  }
}
```

#### If using MkDocs (mkdocs.yml):

```yaml
nav:
  - Home: index.md
  - Explore:
    - Technology:
      - Overview: explore/technology/index.md
      - SpikingBrain-7B: explore/technology/spikingbrain.md
```

### Step 3: Test Locally

```bash
# Depending on your setup:

# Jekyll
bundle exec jekyll serve

# Hugo
hugo server

# VuePress
npm run docs:dev

# MkDocs
mkdocs serve

# Then visit: http://localhost:XXXX/explore/technology/spikingbrain
```

---

## 📋 Integration Checklist

### Pre-Integration
- [ ] Identify your site generator (Jekyll/Hugo/VuePress/etc.)
- [ ] Locate your content directory (docs/, content/, etc.)
- [ ] Find your navigation config file
- [ ] Backup your current site

### During Integration
- [ ] Copy explore-spikingbrain.md to correct location
- [ ] Update navigation/menu configuration
- [ ] Add any custom styling (optional)
- [ ] Test links to external resources

### Post-Integration
- [ ] Test all internal links
- [ ] Verify external links to GitHub/ModelScope
- [ ] Check mobile responsiveness
- [ ] Optimize images (if added)
- [ ] Test on staging environment
- [ ] Deploy to production

---

## 🎨 Customization Options

### 1. Adjust Styling

Add custom CSS for the technology page:

```css
/* In your site's CSS file */
.technology-page {
  max-width: 900px;
  margin: 0 auto;
}

.spike-demo {
  background: #f5f5f5;
  padding: 1rem;
  border-radius: 8px;
  font-family: monospace;
}

.performance-table {
  border-collapse: collapse;
  width: 100%;
}

.highlight-recommendation {
  background: #fffacd;
  border-left: 4px solid #ffa500;
  padding: 0.5rem;
}
```

### 2. Add Custom Headers/Footers

```markdown
<!-- At the top of explore-spikingbrain.md -->
---
layout: technology-page
title: "SpikingBrain-7B Neuromorphic AI"
category: Neuromorphic Computing
tags: [AI, Spiking Neural Networks, Energy Efficiency]
author: Living Intelligence Team
date: 2025-11-25
---

<!-- Your content starts here -->
```

### 3. Enable Comments

If using Disqus, Utterances, or similar:

```markdown
<!-- At the bottom of explore-spikingbrain.md -->

---

## 💬 Discussion

<div class="comments-section">
  <!-- Your comment system here -->
</div>
```

### 4. Add Social Sharing

```html
<!-- Add social sharing buttons -->
<div class="social-share">
  <a href="https://twitter.com/share?text=Exploring SpikingBrain-7B&url=...">
    Share on Twitter
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=...">
    Share on LinkedIn
  </a>
</div>
```

---

## 🔗 Link Structure

### Internal Links (within living-intelligence)

Update these placeholders if needed:

```markdown
<!-- Example internal links -->
[Back to Explore](/explore/)
[Other Technologies](/explore/technology/)
[About Us](/about/)
```

### External Links (to SpikingBrain-7B repo)

Already configured in the page:
- GitHub repository
- Documentation
- Demos
- Model weights

---

## 📱 Responsive Design Tips

The markdown content is already optimized for responsiveness, but you may want to add:

```css
/* Mobile-friendly tables */
@media (max-width: 768px) {
  table {
    display: block;
    overflow-x: auto;
  }
}

/* Mobile-friendly code blocks */
@media (max-width: 768px) {
  pre {
    font-size: 12px;
    padding: 0.5rem;
  }
}
```

---

## 🔍 SEO Optimization

Add metadata to the frontmatter:

```yaml
---
title: "SpikingBrain-7B: Energy-Efficient Neuromorphic AI"
description: "Explore SpikingBrain-7B, a 7-billion parameter AI model with 69% sparsity and 100× energy efficiency through brain-inspired spiking neural networks."
keywords: "neuromorphic AI, spiking neural networks, energy efficient AI, SpikingBrain, living intelligence"
og:image: "/assets/images/spikingbrain-social.png"
og:type: "article"
twitter:card: "summary_large_image"
---
```

---

## 🌐 Multi-Language Support

If your site supports multiple languages:

```
docs/
├── en/
│   └── explore/
│       └── technology/
│           └── spikingbrain.md
├── zh/
│   └── explore/
│       └── technology/
│           └── spikingbrain.md  # Chinese version
└── es/
    └── explore/
        └── technology/
            └── spikingbrain.md  # Spanish version
```

---

## 🧪 Testing Checklist

### Functionality Tests
- [ ] All internal links work
- [ ] All external links work
- [ ] Code snippets are properly formatted
- [ ] Tables render correctly
- [ ] Images load (if added)

### Cross-Browser Tests
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers

### Performance Tests
- [ ] Page load time < 3 seconds
- [ ] Lighthouse score > 90
- [ ] No broken images/links
- [ ] Proper caching headers

---

## 📊 Analytics Integration

### Google Analytics

```html
<!-- Add to your site's head -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Track Specific Events

```javascript
// Track demo link clicks
document.querySelectorAll('a[href*="simple_spike_demo"]').forEach(link => {
  link.addEventListener('click', () => {
    gtag('event', 'demo_click', {
      'event_category': 'engagement',
      'event_label': 'spike_demo'
    });
  });
});
```

---

## 🚨 Troubleshooting

### Issue: Page doesn't appear in navigation

**Solution:** Check your navigation config file and ensure:
1. File path is correct
2. Syntax is valid (YAML/TOML/JSON)
3. Site has been rebuilt after changes

### Issue: Links are broken

**Solution:**
1. Verify base URL configuration
2. Check relative vs. absolute paths
3. Ensure linked files exist

### Issue: Code blocks not highlighting

**Solution:**
1. Install syntax highlighter (Prism, Highlight.js)
2. Add language identifiers to code blocks
3. Include CSS for highlighting

### Issue: Images not loading

**Solution:**
1. Check image paths (relative to page location)
2. Verify images are committed to repo
3. Check file permissions

---

## 🎯 Alternative Site Structures

### Option A: Flat Structure

```
docs/
├── spikingbrain.md
├── other-tech.md
└── index.md
```

Navigation:
```
Home → SpikingBrain-7B
```

### Option B: Category Structure

```
docs/
├── ai-technologies/
│   └── spikingbrain.md
├── hardware/
└── software/
```

Navigation:
```
Home → AI Technologies → SpikingBrain-7B
```

### Option C: Deep Hierarchy

```
docs/
├── explore/
│   ├── cutting-edge/
│   │   └── neuromorphic/
│   │       └── spikingbrain.md
```

Navigation:
```
Home → Explore → Cutting Edge → Neuromorphic → SpikingBrain-7B
```

---

## 📞 Getting Help

### Common Questions

**Q: Can I modify the content?**
A: Yes! The content is provided as a starting point. Customize it to fit your site's style and audience.

**Q: How do I update the content when SpikingBrain-7B releases new features?**
A: Check the main SpikingBrain-7B repository for updates and sync relevant changes.

**Q: Can I use this content in multiple languages?**
A: Yes! Feel free to translate. The Chinese technical report is available in the main repo.

**Q: What license applies?**
A: Same as SpikingBrain-7B (Apache 2.0). Attribution appreciated but not required.

---

## 🎓 Next Steps

After integration:

1. **Announce it!**
   - Social media posts
   - Newsletter announcement
   - Blog post about neuromorphic AI

2. **Add related content**
   - Other neuromorphic technologies
   - Energy-efficient AI techniques
   - Brain-inspired computing

3. **Engage the community**
   - Enable comments/discussions
   - Host webinars or demos
   - Collect feedback

4. **Keep updated**
   - Watch SpikingBrain-7B repo for updates
   - Add new benchmarks/results as available
   - Update with community contributions

---

## ✅ Integration Complete!

Once you've followed these steps, your living-intelligence repository will have a comprehensive "Explore Technology" page showcasing SpikingBrain-7B's neuromorphic AI capabilities.

### Quick Links After Integration

- **View Live Page:** `https://your-site.com/explore/technology/spikingbrain`
- **Edit Content:** `docs/explore/technology/spikingbrain.md`
- **Update Navigation:** Your config file (see Step 2 above)

---

## 📧 Support

Need help with integration?
- Check the main repo's documentation
- Open an issue on GitHub
- Contact via neuronchip.org

---

*Happy integrating! 🚀*
