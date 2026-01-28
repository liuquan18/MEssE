# Testing Guide - Enhanced Monitoring Dashboard

## 🚀 Quick Start

### 1. Start the Monitoring Server

```bash
cd /work/mh1498/m301257/work/MEssE/scripts/monitor
bash start_monitor.sh
```

Expected output:
```
Activating virtual environment...
Starting monitoring server on port 5001...
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://0.0.0.0:5001
```

### 2. Access the Dashboard

**Option A: SSH Port Forwarding (Recommended)**
```bash
# On your local machine:
ssh -L 5001:localhost:5001 your_username@levante.dkrz.de

# Then open in browser:
http://localhost:5001
```

**Option B: Direct Network Access**
```
http://levante.dkrz.de:5001
(May require firewall/network configuration)
```

---

## ✅ Verification Checklist

### Visual Elements

- [ ] **Background Gradient**: Should see blue-to-purple gradient background
- [ ] **Header**: Contains MEssE logo, title, subtitle, and status indicator
- [ ] **Status Indicator**: 
  - 🟢 Green when job is running
  - 🟡 Yellow when waiting
  - 🔴 Red when stopped/error
- [ ] **Animations**: 
  - Header slides down on load
  - Status indicator pulses/glows
  - Cards fade in from bottom
- [ ] **Two-Column Layout**: 
  - Left panel: Simulation + Architecture
  - Right panel: Training Dashboard

### Simulation Panel (Left Top)

- [ ] Shows simulation details with emoji icons
- [ ] Data includes:
  - 📅 Simulation Date
  - ⏰ Start Time
  - ⏱️ Duration (formatted like "5h 23m 45s")
  - 📁 Data Directory
  - 🎯 Timesteps Processed
  - 📍 Current Timestep
- [ ] Cards have subtle hover effect (slide right, highlight border)

### GNN Architecture Panel (Left Bottom)

- [ ] Purple gradient background
- [ ] 2-column grid layout
- [ ] Shows:
  - 🧠 Model Type (e.g., "GNN (Mini-batch)")
  - 📚 Learning Rate
  - 📦 Batch Size
  - 🔄 Batches per Timestep
- [ ] Matches color theme with training panel

### Training Dashboard (Right Panel)

#### Metrics Grid (Top)
- [ ] **4 metric cards** in 2x2 grid:
  1. Current Loss
  2. Average Loss
  3. Minimum Loss
  4. Maximum Loss
- [ ] Each card shows:
  - Label at top
  - Large value in scientific notation
  - Trend indicator/description at bottom
- [ ] Cards have gradient background with shimmer effect
- [ ] Hover effect: lift up and expand shadow

#### Statistics Bar (Middle)
- [ ] **4 stat items** in horizontal row:
  1. Total Batches
  2. Batches/Step
  3. Learning Rate
  4. Model Type (GNN/MLP)
- [ ] Yellow theme background
- [ ] Values on top, labels below

#### Loss Chart (Bottom)
- [ ] Canvas area ~400px height
- [ ] Chart title: "Training Loss Evolution"
- [ ] Purple gradient line (purple → blue)
- [ ] Points visible on hover
- [ ] X-axis: Batch Number
- [ ] Y-axis: Loss Value (scientific notation)
- [ ] Tooltip on hover shows:
  - "Batch X"
  - "Loss: Y.YYe-02"
- [ ] Smooth curve animation

#### Timestamp Footer
- [ ] Shows "Real-time monitoring • Updated: HH:MM:SS"
- [ ] Updates every 2 seconds

---

## 🧪 Functional Testing

### Data Updates

1. **Initial Load**
   - [ ] All panels show "Loading..." spinner initially
   - [ ] Spinner is centered with rotating animation
   - [ ] Data populates within 2 seconds

2. **Real-time Updates**
   - [ ] Status indicator changes based on job state
   - [ ] Timestamp updates every 2 seconds
   - [ ] Loss chart adds new points as training progresses
   - [ ] Metric values update automatically
   - [ ] No page refresh required

3. **Data Formatting**
   - [ ] Duration shows as "Xh Ym Zs" or "Ym Zs" or "Zs"
   - [ ] Large numbers show as "1.5M" or "37.5K"
   - [ ] Scientific notation for loss values: "2.65e-02"
   - [ ] Dates formatted correctly: "2021-07-14"
   - [ ] Times formatted correctly: "00:00:00"

### Edge Cases

1. **No Data**
   - [ ] Gracefully shows "N/A" or "—" for missing values
   - [ ] No JavaScript errors in console
   - [ ] Panels still render correctly

2. **Single Data Point**
   - [ ] Chart displays correctly with one point
   - [ ] Metrics calculate without errors
   - [ ] No division by zero errors

3. **Large Dataset**
   - [ ] Chart handles 100+ data points smoothly
   - [ ] No performance degradation
   - [ ] Chart still readable (uses maxTicksLimit)

### Interactive Elements

1. **Hover Effects**
   - [ ] Info cards highlight on hover
   - [ ] Metric cards lift and glow
   - [ ] Chart points enlarge on hover
   - [ ] Cursor changes appropriately

2. **Chart Interactions**
   - [ ] Tooltip follows cursor
   - [ ] Can hover over any point
   - [ ] Legend items visible
   - [ ] Smooth animations on update

---

## 📱 Responsive Testing

### Desktop (> 1200px)
- [ ] Two-column layout displays correctly
- [ ] All content visible without scrolling horizontally
- [ ] Adequate spacing between elements
- [ ] Chart is full width within panel

### Laptop (1200px - 768px)
- [ ] Layout switches to single column
- [ ] Panels stack vertically
- [ ] Font sizes adjust appropriately
- [ ] No overlapping elements

### Tablet/Mobile (< 768px)
- [ ] Single column layout
- [ ] Metrics grid becomes 1 column
- [ ] Stats bar stacks vertically
- [ ] Chart scales to fit screen
- [ ] Touch interactions work

---

## 🐛 Debugging

### Browser Console

Open Developer Tools (F12) and check:

1. **No JavaScript Errors**
   ```
   ❌ BAD:  Uncaught TypeError: Cannot read property...
   ✅ GOOD: No errors in console
   ```

2. **Network Requests**
   ```
   Check every 2 seconds:
   GET /status → 200 OK
   Response: {"simulation": {...}, "training": {...}, ...}
   ```

3. **Chart.js Load**
   ```
   ✅ Chart.js loaded successfully
   ✅ No CORS errors
   ```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| White screen | Server not running | Start with `bash start_monitor.sh` |
| "Loading..." stuck | `/status` endpoint error | Check server logs |
| Chart not rendering | Chart.js not loaded | Check CDN availability |
| Animations jerky | Browser performance | Try in Chrome/Firefox |
| Colors wrong | Browser cache | Hard refresh (Ctrl+Shift+R) |
| Port 5001 in use | Another process | `lsof -i :5001` and kill process |

### Server Logs

Check Flask output:
```bash
# While server is running, you'll see:
127.0.0.1 - - [DATE TIME] "GET /status HTTP/1.1" 200 -
127.0.0.1 - - [DATE TIME] "GET / HTTP/1.1" 200 -
```

Look for errors:
```
❌ 500 Internal Server Error → Check status.json file
❌ 404 Not Found → Check file paths in app.py
```

---

## 📊 Performance Benchmarks

### Expected Performance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Initial Load Time | < 1s | ___ | ⏱️ |
| Status Update Interval | 2s | 2s | ✅ |
| Chart Animation | 750ms | 750ms | ✅ |
| Memory Usage (browser) | < 100 MB | ___ | ⏱️ |
| CPU Usage (browser) | < 5% | ___ | ⏱️ |

### Data Volume Handling

| Dataset Size | Expected Behavior | Status |
|--------------|-------------------|--------|
| 0-10 points | Instant render | ___ |
| 10-100 points | < 1s render | ___ |
| 100-1000 points | < 2s render | ___ |
| 1000+ points | May slow down, consider pagination | ___ |

---

## 🎨 Visual Regression Testing

### Color Accuracy

Take screenshots and verify:

1. **Background Gradient**
   - Start: `#1e3a8a` (dark blue)
   - Middle: `#3b82f6` (blue)
   - End: `#8b5cf6` (purple)

2. **Metric Cards**
   - Gradient: `#3b82f6` → `#8b5cf6`
   - Text: White (`#ffffff`)
   - Shadow: `rgba(59, 130, 246, 0.3)`

3. **Chart Colors**
   - Line: `rgba(147, 51, 234, 1)` (purple)
   - Fill: Purple to blue gradient
   - Grid: `rgba(255, 255, 255, 0.1)`
   - Text: `#cbd5e1` (slate)

### Typography

- [ ] **Headers**: Inter 700 (bold)
- [ ] **Body**: Inter 400 (regular)
- [ ] **Metrics**: Inter 800 (extra bold)
- [ ] **Code**: SF Mono (monospace)

### Spacing

- [ ] Panel padding: 24px
- [ ] Gap between cards: 16px
- [ ] Margin between sections: 24px
- [ ] Grid gaps: 16px

---

## 🔬 Advanced Testing

### Stress Testing

1. **Rapid Updates**
   - Generate status.json with updates every 100ms
   - Check if UI remains responsive
   - Monitor memory leaks

2. **Large Datasets**
   - Create loss array with 10,000 points
   - Verify chart renders without crashing
   - Check performance metrics

3. **Long Running**
   - Leave dashboard open for 1+ hour
   - Check for memory leaks
   - Verify no UI degradation

### Cross-Browser Testing

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| Chrome | Latest | ___ | Recommended |
| Firefox | Latest | ___ | Good support |
| Safari | Latest | ___ | May need testing |
| Edge | Latest | ___ | Chromium-based |

---

## 📝 Test Report Template

```markdown
## Test Session Report

**Date**: YYYY-MM-DD
**Tester**: [Name]
**Browser**: [Browser + Version]
**Screen Size**: [Resolution]

### Passed Tests
- [ ] Item 1
- [ ] Item 2

### Failed Tests
- [ ] Item 3 - [Description of failure]

### Performance
- Load Time: ___
- Memory Usage: ___
- CPU Usage: ___

### Screenshots
[Attach screenshots of any issues]

### Notes
[Additional observations]
```

---

## 🎯 Acceptance Criteria

### Must Have ✅
- [ ] Dashboard loads without errors
- [ ] Status updates every 2 seconds
- [ ] All panels display data correctly
- [ ] Chart renders and updates
- [ ] Responsive design works
- [ ] No console errors
- [ ] Animations are smooth

### Nice to Have 🌟
- [ ] Hover effects work perfectly
- [ ] Loading states are elegant
- [ ] Colors match design spec exactly
- [ ] Performance is excellent
- [ ] Works in all major browsers

---

## 📞 Support

If you encounter issues:

1. **Check Server Logs**: Look for Python errors
2. **Check Browser Console**: Look for JavaScript errors
3. **Check Network Tab**: Verify `/status` endpoint
4. **Check Files**:
   - `app.py` - Flask server
   - `templates/monitor.html` - UI file
   - `status.json` - Data file
   - `start_monitor.sh` - Launch script

---

## 🎉 Success Indicators

You know the dashboard is working perfectly when:

1. ✨ Beautiful gradient background
2. 📊 Live data updates every 2 seconds
3. 📈 Smooth chart animations
4. 🎨 All colors and shadows render correctly
5. 🖱️ Hover effects are responsive and elegant
6. 📱 Works on different screen sizes
7. 🚀 Performance is snappy
8. 😊 You enjoy looking at it!

---

**Happy Monitoring! 🎊**
